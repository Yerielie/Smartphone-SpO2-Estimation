"""Section 3: CNN SpO2 regression on the Hoffman et al. (2022) data, leave-one-subject-out.

From the paper's description:
  * input = one 3-s window of R, G, B (3 x 90 frames), output = SpO2 (regression)
  * 3 convolutional layers + 2 fully connected layers, ReLU, kernel size 3, no padding
  * MSE loss, Adam, lr 1e-5 decayed x0.1 after 80 epochs, L2 regularisation (weight decay) 0.1
  * LOSO: 1 test subject, 1 validation subject, 4 training subjects per fold

Our own choices (the paper does not specify them; see results/section3_summary.md):
  * 1D convolutions over time with R, G, B as input channels; 8 / 16 / 16 filters; no pooling
  * hidden FC layer of 32 units
  * input: per-channel z-score with mean / SD of all training frames (training subjects only)
  * target: z-scored with the training-set SpO2 mean / SD (predictions are mapped back to %)
  * batch size 64, 100 epochs, model selection = epoch with the lowest validation MSE
  * validation subject = the next subject in SUBJECTS after the test subject (cyclic)
  * Adam's built-in (coupled) weight_decay is used as the L2 term
  * 3 seeds (0, 1, 2): different weight init and batch order, same splits

Run from the repository root:  python src/public_data_spo2_cnn.py
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset import FPS, SUBJECTS, build_windows
# Same metric functions and plot style as Section 2
from public_data_spo2_hr_calibrated_formula import (INK, INK2, REF, SERIES,
                                                    _style, all_metrics)

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = [0, 1, 2]
WINDOW_SEC = 3.0
EPOCHS = 100
BATCH = 64
LR, LR_STEP, LR_GAMMA = 1e-5, 80, 0.1
WEIGHT_DECAY = 0.1
FILTERS = (8, 16, 16)
HIDDEN = 32
SERIES2 = "#eb6834"   # second categorical slot (validation curve)


# ---------------------------------------------------------------- model
class SpO2CNN(nn.Module):
    """3 x Conv1d(k=3, no padding) + ReLU, then 2 fully connected layers."""

    def __init__(self, n_in=3, win=90, filters=FILTERS, hidden=HIDDEN):
        super().__init__()
        layers, c = [], n_in
        for f in filters:
            layers += [nn.Conv1d(c, f, kernel_size=3, padding=0), nn.ReLU()]
            c = f
        self.conv = nn.Sequential(*layers)
        n_flat = c * (win - 2 * len(filters))   # each unpadded k=3 conv drops 2 frames
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))

    def forward(self, x):            # x: (batch, 3, 90)
        return self.fc(self.conv(x)).squeeze(-1)


# ---------------------------------------------------------------- normalisation
def fit_norm(X, y):
    """Per-channel mean / SD over all frames, and target mean / SD (training data only)."""
    return {"x_mean": X.mean(axis=(0, 2)), "x_std": X.std(axis=(0, 2)),
            "y_mean": float(y.mean()), "y_std": float(y.std())}


def apply_norm(X, norm):
    return ((X - norm["x_mean"][None, :, None]) / norm["x_std"][None, :, None]).astype(np.float32)


# ---------------------------------------------------------------- training
def predict(model, X, norm):
    """SpO2 in % for normalised inputs X (tensor)."""
    model.eval()
    with torch.no_grad():
        z = torch.cat([model(X[i:i + 4096]) for i in range(0, len(X), 4096)])
    return z.numpy() * norm["y_std"] + norm["y_mean"]


def train(X_tr, y_tr, X_val, y_val, norm, seed, epochs=EPOCHS):
    """Train one model. Returns (model with best-validation weights, history, best epoch).

    If X_val is None (final model), no selection is done and the last epoch is kept.
    Losses in the history are MSE in SpO2 %^2 units.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = SpO2CNN(win=X_tr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(apply_norm(X_tr, norm))
    zt = torch.from_numpy(((y_tr - norm["y_mean"]) / norm["y_std"]).astype(np.float32))
    Xv = torch.from_numpy(apply_norm(X_val, norm)) if X_val is not None else None

    hist, best = [], (np.inf, 0, None)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tot = 0.0
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), zt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        row = {"epoch": ep, "train_mse": tot / len(Xt) * norm["y_std"] ** 2}
        if Xv is not None:
            row["val_mse"] = float(np.mean((predict(model, Xv, norm) - y_val) ** 2))
            if row["val_mse"] < best[0]:
                best = (row["val_mse"], ep, {k: v.clone() for k, v in model.state_dict().items()})
        hist.append(row)
    if Xv is not None:
        model.load_state_dict(best[2])
        return model, hist, best[1]
    return model, hist, epochs


# ---------------------------------------------------------------- evaluation
def metrics_row(y, p):
    m = all_metrics(y, p)
    m["Arms"] = float(np.sqrt(np.mean((p - y) ** 2)))   # ISO 80601-2-61 Arms (= RMSE vs reference)
    return m


def run_loso(X, y, subj):
    """Every fold x seed. Returns predictions (seed -> array), loss history rows, fold info rows."""
    preds = {s: np.full_like(y, np.nan) for s in SEEDS}
    hist_rows, fold_rows = [], []
    for k, test_s in enumerate(SUBJECTS):
        val_s = SUBJECTS[(k + 1) % len(SUBJECTS)]
        te, va = subj == test_s, subj == val_s
        tr = ~te & ~va
        norm = fit_norm(X[tr], y[tr])
        for seed in SEEDS:
            t0 = time.time()
            model, hist, best_ep = train(X[tr], y[tr], X[va], y[va], norm, seed)
            preds[seed][te] = predict(model, torch.from_numpy(apply_norm(X[te], norm)), norm)
            hist_rows += [{"test_subject": test_s, "val_subject": val_s, "seed": seed, **h} for h in hist]
            fold_rows.append({"test_subject": test_s, "val_subject": val_s, "seed": seed,
                              "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                              "best_epoch": best_ep,
                              "best_val_mse": min(h["val_mse"] for h in hist)})
            mae = np.mean(np.abs(preds[seed][te] - y[te]))
            print(f"test {test_s} val {val_s} seed {seed}: best epoch {best_ep}, "
                  f"test MAE {mae:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    return preds, pd.DataFrame(hist_rows), pd.DataFrame(fold_rows)


def summarise(y, subj, preds):
    """Per-seed metrics (per subject, folds mean / sd, pooled), then mean / sd across seeds."""
    rows = []
    for seed, p in preds.items():
        fold = []
        for s in SUBJECTS:
            m = subj == s
            met = metrics_row(y[m], p[m])
            fold.append(met)
            rows.append({"seed": seed, "scope": f"subject {s}", **met})
        fd = pd.DataFrame(fold)
        rows.append({"seed": seed, "scope": "folds mean", **fd.mean().to_dict()})
        rows.append({"seed": seed, "scope": "folds sd", **fd.std(ddof=1).to_dict()})
        rows.append({"seed": seed, "scope": "pooled", **metrics_row(y, p)})
    per_seed = pd.DataFrame(rows)
    cols = [c for c in per_seed.columns if c not in ("seed", "scope")]
    across = []
    for scope, g in per_seed.groupby("scope", sort=False):
        across.append({"seed": "mean over seeds", "scope": scope, **g[cols].mean().to_dict()})
        across.append({"seed": "sd over seeds", "scope": scope, **g[cols].std(ddof=1).to_dict()})
    return pd.concat([per_seed, pd.DataFrame(across)], ignore_index=True)


# ---------------------------------------------------------------- plots
def plot_scatter(y, p, met, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(y, p, s=8, color=SERIES, alpha=0.15, linewidths=0)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color=REF, linewidth=1.2, linestyle="--", label="identity")
    ax.set_xlim(lo - 2, hi + 2); ax.set_ylim(lo - 2, hi + 2)
    ax.set_title(f"CNN, LOSO test predictions (seed 0, pooled)\n"
                 f"MAE {met['MAE']:.2f}, R² {met['R2']:.2f}", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Ground-truth SpO2 (%)", color=INK2)
    ax.set_ylabel("Predicted SpO2 (%)", color=INK2)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_bland_altman(y, p, path):
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
    ax.set_title("Bland-Altman: CNN (seed 0, pooled LOSO test predictions)", fontsize=10,
                 color=INK, loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_losses(hist, folds, path):
    """Train / validation MSE per epoch for each fold; thin line per seed."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, SUBJECTS):
        for seed in SEEDS:
            h = hist[(hist.test_subject == s) & (hist.seed == seed)]
            lab = seed == SEEDS[0]
            ax.plot(h.epoch, h.train_mse, color=SERIES, linewidth=1.2, alpha=0.8,
                    label="train (4 subjects)" if lab else None)
            ax.plot(h.epoch, h.val_mse, color=SERIES2, linewidth=1.2, alpha=0.8,
                    label="validation (1 subject)" if lab else None)
        be = folds[folds.test_subject == s].best_epoch.to_numpy()
        val_s = folds[folds.test_subject == s].val_subject.iloc[0]
        ax.axvline(LR_STEP, color=REF, linewidth=1, linestyle=":")
        ax.set_title(f"test {s}, val {val_s}\nbest epochs {', '.join(map(str, be))}",
                     fontsize=9, color=INK, loc="left")
        ax.set_yscale("log")
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Epoch", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (SpO2 %²)", color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("CNN loss curves per LOSO fold (3 seeds each; dotted line = LR decay at epoch 80)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    RESULTS.mkdir(exist_ok=True)
    d = build_windows(window_sec=WINDOW_SEC)
    X = np.stack([d["r"], d["g"], d["b"]], axis=1).astype(np.float64)   # (n, 3, 90)
    y, subj = d["spo2"].astype(np.float64), d["subject"]
    print(f"3-s windows: {len(y)}  input shape {X.shape}")

    preds, hist, folds = run_loso(X, y, subj)
    metrics = summarise(y, subj, preds)
    metrics.to_csv(RESULTS / "section3_metrics.csv", index=False)
    folds.to_csv(RESULTS / "section3_folds.csv", index=False)
    hist.to_csv(RESULTS / "section3_loss_history.csv", index=False)
    # Window-level predictions are not saved: they would repeat the dataset's ground truth,
    # and data is not committed to this repository.

    # Per-subject table: mean +/- sd over seeds
    keys = ["MAE", "RMSE", "R2", "bias", "sensitivity", "specificity"]
    ps = metrics[metrics.seed.isin(SEEDS) & metrics.scope.str.startswith("subject")]
    per_subj = ps.groupby("scope")[keys + ["n", "n_pos", "n_neg"]].agg(["mean", "std"])
    per_subj.columns = [f"{a}_{b}" for a, b in per_subj.columns]
    per_subj = per_subj.drop(columns=["n_std", "n_pos_std", "n_neg_std"])
    per_subj.index = per_subj.index.str.replace("subject ", "")
    per_subj.index.name = "subject"
    per_subj.to_csv(RESULTS / "section3_per_subject.csv")

    p0 = preds[SEEDS[0]]
    plot_scatter(y, p0, metrics_row(y, p0), RESULTS / "section3_scatter.png")
    plot_bland_altman(y, p0, RESULTS / "section3_bland_altman.png")
    plot_losses(hist, folds, RESULTS / "section3_loss_curves.png")

    # Final model on all 6 subjects. No held-out subject is left for early stopping, so the
    # number of epochs is the median best epoch over the 18 LOSO runs (chosen on validation only).
    final_epochs = int(np.median(folds.best_epoch))
    norm = fit_norm(X, y)
    model, hist_final, _ = train(X, y, None, None, norm, seed=SEEDS[0], epochs=final_epochs)
    torch.save({"state_dict": model.state_dict(),
                "x_mean": torch.tensor(norm["x_mean"]), "x_std": torch.tensor(norm["x_std"]),
                "y_mean": norm["y_mean"], "y_std": norm["y_std"],
                "filters": FILTERS, "hidden": HIDDEN, "window_frames": X.shape[2]},
               RESULTS / "section3_model.pt")
    info = {
        "model_file": "section3_model.pt",
        "architecture": {"class": "SpO2CNN in src/public_data_spo2_cnn.py",
                         "conv1d_filters": list(FILTERS), "kernel_size": 3, "padding": 0,
                         "fc_hidden": HIDDEN, "activation": "ReLU", "output": "SpO2 (%)"},
        "input": {"shape": [3, X.shape[2]], "channel_order": ["R", "G", "B"],
                  "values": "mean pixel value per frame over the fingertip image, raw (0-255 scale)",
                  "fs_hz": FPS, "window_sec": WINDOW_SEC,
                  "recording_conditions_of_training_data": "Nexus 6P, flash on, locked exposure, "
                  "no auto white balance, fixed channel gains R 1x / G 3x / B 18x"},
        "preprocessing": {"x_norm": "(x - x_mean[c]) / x_std[c] per channel c",
                          "x_mean": norm["x_mean"].tolist(), "x_std": norm["x_std"].tolist(),
                          "filtering": "none (raw frame means, no band-pass)"},
        "output_denormalisation": {"formula": "spo2 = model_output * y_std + y_mean",
                                   "y_mean": norm["y_mean"], "y_std": norm["y_std"]},
        "training": {"subjects": SUBJECTS, "n_windows": int(len(y)), "seed": SEEDS[0],
                     "epochs": final_epochs, "epochs_rule": "median best-validation epoch of the LOSO runs",
                     "optimizer": "Adam", "lr": LR, "lr_decay": {"after_epoch": LR_STEP, "gamma": LR_GAMMA},
                     "weight_decay": WEIGHT_DECAY, "batch_size": BATCH, "loss": "MSE on z-scored SpO2",
                     "final_train_mse": hist_final[-1]["train_mse"],
                     "gt": "Masimo Radical-7 (SpO2 5)", "excluded": "GT SpO2 < 70 or missing"},
    }
    (RESULTS / "section3_model.json").write_text(json.dumps(info, indent=2))

    pd.set_option("display.width", 220)
    show = ["seed", "scope", "MAE", "RMSE", "Arms", "R2", "bias", "sensitivity", "specificity"]
    print(metrics[~metrics.scope.str.startswith("subject")][show].to_string(index=False))
    print(per_subj.round(3).to_string())
    print(folds.to_string(index=False))
    print(f"final model: {final_epochs} epochs, train MSE {hist_final[-1]['train_mse']:.2f}")


if __name__ == "__main__":
    main()
