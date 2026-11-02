"""Part 1: classical ratio-of-ratios SpO2 + peak-based heart rate on the Hoffman et al. (2022) data.

Signal processing mirrors src/spo2_hr_from_video.py (SpO2Estimator):
  * Butterworth band-pass, order 1, 0.67-4.0 Hz, filtfilt
  * AC = std of the filtered window, DC = mean of the raw window
  * R  = (AC_red / DC_red) / (AC_x / DC_x),   x = blue (reference) or green
  * fixed formula SpO2 = 110 - 25 R (no clamping here, unlike the video script)
  * HR: find_peaks on filtered red (height = std, distance = fs/4), HR = 60 / mean interval

Calibrated model SpO2 = a - b R is evaluated with leave-one-subject-out CV
(both hands of a subject are always in the same fold).

Run from the repository root:  python src/ratio_of_ratios.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from dataset import FPS, SUBJECTS, alignment_report, build_windows

RESULTS = Path(__file__).resolve().parents[1] / "results"
LOW_HZ, HIGH_HZ, ORDER = 0.67, 4.0, 1
HYPOX_THRESHOLD = 90.0
PAIRS = {"red/blue": "b", "red/green": "g"}

# Plot styling (single-series charts: one colour, recessive reference lines)
SERIES = "#2a78d6"
REF = "#8a8984"
INK = "#0b0b0b"
INK2 = "#52514e"


# ---------------------------------------------------------------- signal processing
def bandpass(x, fs=FPS):
    """Zero-phase band-pass along the last axis (works on one window or a stack)."""
    nyq = 0.5 * fs
    b, a = butter(ORDER, [LOW_HZ / nyq, HIGH_HZ / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def ratio_of_ratios(red, other, fs=FPS):
    """R for each row of `red` / `other` (shape (n, window_len))."""
    ac_red = np.std(bandpass(red, fs), axis=-1)
    ac_oth = np.std(bandpass(other, fs), axis=-1)
    dc_red = np.mean(red, axis=-1)
    dc_oth = np.mean(other, axis=-1)
    return (ac_red / dc_red) / (ac_oth / dc_oth)


def heart_rate(red, fs=FPS):
    """Peak-based HR (bpm) per window; NaN if fewer than two peaks are found."""
    filt = bandpass(red, fs)
    hr = np.full(len(filt), np.nan)
    for k, sig in enumerate(filt):
        peaks, _ = find_peaks(sig, height=np.std(sig), distance=fs / 4)
        if len(peaks) >= 2:
            hr[k] = 60.0 / (np.mean(np.diff(peaks)) / fs)
    return hr


def fixed_formula(R):
    return 110.0 - 25.0 * R


def fit_linear(R, spo2):
    """Least-squares fit of SpO2 = a - b R. Returns (a, b)."""
    slope, intercept = np.polyfit(R, spo2, 1)
    return float(intercept), float(-slope)


# ---------------------------------------------------------------- metrics
def regression_metrics(y, yhat):
    err = yhat - y
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "bias": float(np.mean(err)),
    }


def hypoxemia_metrics(y, yhat, thr=HYPOX_THRESHOLD):
    """Sensitivity / specificity for SpO2 < thr (positive = hypoxemic)."""
    pos, pred = y < thr, yhat < thr
    tp, fn = np.sum(pos & pred), np.sum(pos & ~pred)
    tn, fp = np.sum(~pos & ~pred), np.sum(~pos & pred)
    return {
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
    }


def all_metrics(y, yhat):
    return {**regression_metrics(y, yhat), **hypoxemia_metrics(y, yhat)}


# ---------------------------------------------------------------- evaluation
def evaluate_spo2(data):
    """Fixed + LOSO-calibrated predictions for every channel pair.

    Returns (rows for metrics table, predictions dict, per-fold coefficients).
    """
    y, subj = data["spo2"], data["subject"]
    rows, preds, fold_coefs = [], {}, []
    for pair, ch in PAIRS.items():
        R = ratio_of_ratios(data["r"], data[ch])
        n_bad = int(np.sum(~np.isfinite(R)))
        if n_bad:
            raise ValueError(f"{pair}: {n_bad} windows with non-finite R")

        # (b) uncalibrated reference
        p_fixed = fixed_formula(R)
        # (c) leave-one-subject-out calibration
        p_cal = np.full_like(y, np.nan)
        for s in SUBJECTS:
            test = subj == s
            a, b = fit_linear(R[~test], y[~test])
            p_cal[test] = a - b * R[test]
            fold_coefs.append({"pair": pair, "test_subject": s, "a": a, "b": b})

        for method, p in [("fixed 110-25R", p_fixed), ("calibrated LOSO", p_cal)]:
            model = f"{pair} | {method}"
            preds[model] = {"pred": p, "R": R}
            fold = []
            for s in SUBJECTS:
                m = subj == s
                met = all_metrics(y[m], p[m])
                fold.append(met)
                rows.append({"task": "spo2", "model": model, "scope": f"subject {s}", **met})
            fold_df = pd.DataFrame(fold)
            for stat in ["mean", "sd"]:
                agg = fold_df.mean() if stat == "mean" else fold_df.std(ddof=1)
                rows.append({"task": "spo2", "model": model, "scope": f"folds {stat}",
                             **{k: float(agg[k]) for k in fold_df.columns}})
            rows.append({"task": "spo2", "model": model, "scope": "pooled", **all_metrics(y, p)})
    return rows, preds, fold_coefs


def evaluate_hr(win3, win10):
    """Compare peak-based HR from 3-s and 10-s windows with GT pulse on common samples."""
    key = lambda d: pd.MultiIndex.from_arrays([d["subject"], d["hand"], d["gt_index"]])
    hr3 = pd.Series(heart_rate(win3["r"]), index=key(win3))
    hr10 = pd.Series(heart_rate(win10["r"]), index=key(win10))
    gt = pd.Series(win3["pulse"], index=key(win3))

    common = hr3.index.intersection(hr10.index)          # windows that exist for both lengths
    common = common[gt.loc[common].notna().to_numpy()]   # ...and have a GT pulse
    rows, out = [], {"gt": gt.loc[common].to_numpy()}
    for name, hr in [("HR 3-s window", hr3), ("HR 10-s window", hr10)]:
        est = hr.loc[common].to_numpy()
        ok = np.isfinite(est)
        g, e = out["gt"][ok], est[ok]
        out[name] = est
        rows.append({
            "task": "hr", "model": name, "scope": "pooled",
            "n": int(ok.sum()), "coverage": float(ok.mean()),
            "MAE": float(np.mean(np.abs(e - g))),
            "RMSE": float(np.sqrt(np.mean((e - g) ** 2))),
            "bias": float(np.mean(e - g)),
            "pearson_r": float(np.corrcoef(g, e)[0, 1]),
        })
    # Same comparison restricted to windows where both lengths produced an estimate
    both = np.isfinite(out["HR 3-s window"]) & np.isfinite(out["HR 10-s window"])
    for name in ["HR 3-s window", "HR 10-s window"]:
        g, e = out["gt"][both], out[name][both]
        rows.append({
            "task": "hr", "model": name, "scope": "both windows valid",
            "n": int(both.sum()), "coverage": np.nan,
            "MAE": float(np.mean(np.abs(e - g))),
            "RMSE": float(np.sqrt(np.mean((e - g) ** 2))),
            "bias": float(np.mean(e - g)),
            "pearson_r": float(np.corrcoef(g, e)[0, 1]),
        })
    out["subject"] = np.array([c[0] for c in common])
    return rows, out


def per_recording_correlation(data, preds):
    """Diagnostic: Pearson r between R and GT SpO2 within each recording, and median R."""
    rows = []
    for pair in PAIRS:
        R = preds[f"{pair} | fixed 110-25R"]["R"]
        for s in SUBJECTS:
            for h in ["Left", "Right"]:
                m = (data["subject"] == s) & (data["hand"] == h)
                rows.append({"pair": pair, "subject": s, "hand": h, "n": int(m.sum()),
                             "median_R": float(np.median(R[m])),
                             "pearson_r_R_vs_spo2": float(np.corrcoef(R[m], data["spo2"][m])[0, 1])})
        rows.append({"pair": pair, "subject": "all", "hand": "both", "n": int(len(R)),
                     "median_R": float(np.median(R)),
                     "pearson_r_R_vs_spo2": float(np.corrcoef(R, data["spo2"])[0, 1])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def _style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(REF)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_scatter(y, preds, pair, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, method in zip(axes, ["fixed 110-25R", "calibrated LOSO"]):
        model = f"{pair} | {method}"
        p = preds[model]["pred"]
        met = regression_metrics(y, p)
        ax.scatter(y, p, s=8, color=SERIES, alpha=0.15, linewidths=0)
        lo = min(y.min(), p.min()); hi = max(y.max(), p.max())   # full range, nothing clipped
        ax.plot([lo, hi], [lo, hi], color=REF, linewidth=1.2, linestyle="--", label="identity")
        ax.set_ylim(lo - 2, hi + 2)
        ax.set_title(f"{method}\npooled MAE {met['MAE']:.2f}, R² {met['R2']:.2f}",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Ground-truth SpO2 (%)", color=INK2)
        ax.set_ylabel("Predicted SpO2 (%)", color=INK2)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Ratio-of-ratios ({pair}): predicted vs ground truth, 3-s windows",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bland_altman(y, p, model, path):
    mean, diff = (y + p) / 2, p - y
    bias, sd = diff.mean(), diff.std(ddof=1)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(mean, diff, s=8, color=SERIES, alpha=0.15, linewidths=0)
    for val, lab, ls in [(bias, f"bias {bias:.2f}", "-"),
                         (bias + 1.96 * sd, f"+1.96 SD {bias + 1.96 * sd:.2f}", "--"),
                         (bias - 1.96 * sd, f"−1.96 SD {bias - 1.96 * sd:.2f}", "--")]:
        ax.axhline(val, color=REF, linewidth=1.2, linestyle=ls)
        ax.text(mean.max(), val, f" {lab}", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("Mean of predicted and ground-truth SpO2 (%)", color=INK2)
    ax.set_ylabel("Predicted − ground truth (%)", color=INK2)
    ax.set_title(f"Bland-Altman: {model} (pooled LOSO test predictions)", fontsize=10,
                 color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hr(hr_out, hr_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    g = hr_out["gt"]
    for ax, row in zip(axes, hr_rows):
        e = hr_out[row["model"]]
        ok = np.isfinite(e)
        ax.scatter(g[ok], e[ok], s=8, color=SERIES, alpha=0.15, linewidths=0)
        allv = np.concatenate([g, *[hr_out[r["model"]] for r in hr_rows]])
        lo, hi = np.nanmin(allv) - 5, np.nanmax(allv) + 5   # full range, nothing clipped
        ax.plot([lo, hi], [lo, hi], color=REF, linewidth=1.2, linestyle="--", label="identity")
        ax.set_xlim(g.min() - 5, g.max() + 5); ax.set_ylim(lo, hi)
        ax.set_title(f"{row['model']}\nMAE {row['MAE']:.2f} bpm, r {row['pearson_r']:.2f}, "
                     f"coverage {100 * row['coverage']:.1f}%", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Ground-truth pulse, Masimo (bpm)", color=INK2)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Estimated HR, red-channel peaks (bpm)", color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    RESULTS.mkdir(exist_ok=True)

    align = alignment_report()
    align.to_csv(RESULTS / "part1_alignment.csv", index=False)

    win3 = build_windows(window_sec=3.0)
    win10 = build_windows(window_sec=10.0)
    y = win3["spo2"]

    spo2_rows, preds, fold_coefs = evaluate_spo2(win3)
    hr_rows, hr_out = evaluate_hr(win3, win10)

    metrics = pd.DataFrame(spo2_rows + hr_rows)
    metrics.to_csv(RESULTS / "part1_metrics.csv", index=False)
    pd.DataFrame(fold_coefs).to_csv(RESULTS / "part1_fold_coefficients.csv", index=False)
    corr = per_recording_correlation(win3, preds)
    corr.to_csv(RESULTS / "part1_r_correlation.csv", index=False)

    # Best calibrated model = lowest mean MAE across folds
    cal = metrics[(metrics.scope == "folds mean") & metrics.model.str.contains("calibrated")]
    best = cal.loc[cal.MAE.idxmin(), "model"]
    best_pair = best.split(" | ")[0]

    # (f) final coefficients on all subjects
    coefs = {}
    for pair, ch in PAIRS.items():
        a, b = fit_linear(preds[f"{pair} | calibrated LOSO"]["R"], y)
        coefs[pair] = {"a": a, "b": b}
        if b <= 0:   # flag a fit that runs against the expected physiology (SpO2 falls as R rises)
            coefs[pair]["warning"] = ("b <= 0: fitted SpO2 does not decrease with R; "
                                      "the fit is not physiologically meaningful")
    out = {
        "formula": "SpO2 = a - b * R",
        "R_definition": {"red/blue": "(AC_R/DC_R)/(AC_B/DC_B)", "red/green": "(AC_R/DC_R)/(AC_G/DC_G)"},
        "coefficients": coefs,
        "best_pair_by_loso_mae": best_pair,
        "preprocessing": {"bandpass_hz": [LOW_HZ, HIGH_HZ], "butter_order": ORDER, "filter": "filtfilt",
                          "AC": "std of band-passed window", "DC": "mean of raw window",
                          "window_sec": 3.0, "fs_hz": FPS},
        "training_data": {"subjects": SUBJECTS, "hands": ["Left", "Right"], "n_windows": int(len(y)),
                          "gt": "Masimo Radical-7 (SpO2 5)", "excluded": "GT SpO2 < 70 or missing"},
    }
    (RESULTS / "part1_coefficients.json").write_text(json.dumps(out, indent=2))

    for pair in PAIRS:
        fname = pair.replace("/", "_")
        plot_scatter(y, preds, pair, RESULTS / f"part1_scatter_{fname}.png")
    plot_bland_altman(y, preds[best]["pred"], best, RESULTS / "part1_bland_altman_best.png")
    plot_hr(hr_out, hr_rows, RESULTS / "part1_hr_scatter.png")

    # Console summary
    pd.set_option("display.width", 200)
    show = ["model", "scope", "n", "MAE", "RMSE", "R2", "bias", "sensitivity", "specificity"]
    print(align.to_string(index=False))
    print(f"\n3-s windows: {len(y)}, 10-s windows: {len(win10['spo2'])}")
    print(metrics[metrics.task == "spo2"][show].to_string(index=False))
    print(metrics[metrics.task == "hr"][["model", "n", "coverage", "MAE", "RMSE", "bias", "pearson_r"]]
          .to_string(index=False))
    print(corr.to_string(index=False))
    print(pd.DataFrame(fold_coefs).to_string(index=False))
    print(f"GT SpO2 mean {y.mean():.3f}, fraction < 90: {np.mean(y < HYPOX_THRESHOLD):.4f}")
    print("\nbest calibrated:", best)
    print(json.dumps(coefs, indent=2))


if __name__ == "__main__":
    main()
