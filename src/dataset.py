"""Load the Hoffman et al. (2022) public dataset and cut windows around each ground-truth second.

Smartphone PPG: data/raw/data/ppg-csv/{Left,Right}/<subject>.csv  (R,G,B mean pixel value, 30 Hz)
Ground truth:   data/raw/data/gt/<subject>.csv                    (1 Hz, Masimo Radical-7 = "SpO2 5"/"Pulse 5")

The camera and the oximeters were started and stopped together, so ground-truth row i
is aligned to frame i * FPS (alignment by index, no timestamps on the video side).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "data"
SUBJECTS = ["100001", "100002", "100003", "100004", "100005", "100006"]
HANDS = ["Left", "Right"]
FPS = 30
SPO2_COL, PULSE_COL = "SpO2 5", "Pulse 5"   # Masimo Radical-7 (paper's ground truth)
MIN_SPO2 = 70                               # paper excludes ground truth below 70 %


def load_ground_truth(subject):
    """Return a DataFrame with columns spo2, pulse (NaN where the device reported 0)."""
    gt = pd.read_csv(DATA_DIR / "gt" / f"{subject}.csv", encoding="utf-8-sig")
    gt = gt.rename(columns={gt.columns[0]: "Time"})   # header of 100004 has no "Time" label
    # Each file ends with a "Collection Halted" footer row; drop it.
    gt = gt[gt["Time"].astype(str).str.strip() != "Collection Halted"].reset_index(drop=True)
    out = pd.DataFrame({
        "spo2": pd.to_numeric(gt[SPO2_COL], errors="coerce"),
        "pulse": pd.to_numeric(gt[PULSE_COL], errors="coerce"),
    })
    return out.replace(0, np.nan)   # 0 = missing reading


def load_ppg(subject, hand):
    """Return a (3, N) array of R, G, B frame means."""
    df = pd.read_csv(DATA_DIR / "ppg-csv" / hand / f"{subject}.csv")
    return df[["R", "G", "B"]].to_numpy().T


def alignment_report():
    """Frame count vs ground-truth length for every recording (should be ~30 frames per GT row)."""
    rows = []
    for s in SUBJECTS:
        n_gt = len(load_ground_truth(s))
        for h in HANDS:
            n_frames = load_ppg(s, h).shape[1]
            rows.append({
                "subject": s, "hand": h, "gt_rows": n_gt, "frames": n_frames,
                "frames_per_gt_row": n_frames / n_gt,
                "frames_minus_30x_gt": n_frames - FPS * n_gt,
                "drift_seconds": (n_frames - FPS * n_gt) / FPS,
            })
    return pd.DataFrame(rows)


def build_windows(window_sec=3.0, min_spo2=MIN_SPO2):
    """Cut one window of `window_sec` seconds centred on frame i*FPS for every GT row i.

    Windows that would run past either end of the recording are skipped, as are rows
    with missing SpO2 or SpO2 < min_spo2. Missing pulse is kept as NaN.

    Returns a dict:
        r, g, b : (n, window_len) arrays
        spo2, pulse : (n,) ground truth
        subject, hand : (n,) string arrays
        gt_index : (n,) ground-truth row index
    """
    win = int(round(window_sec * FPS))
    half = win // 2
    r, g, b, spo2, pulse, subj, hand, idx = [], [], [], [], [], [], [], []
    for s in SUBJECTS:
        gt = load_ground_truth(s)
        for h in HANDS:
            wave = load_ppg(s, h)
            n_frames = wave.shape[1]
            for i, row in gt.iterrows():
                if np.isnan(row.spo2) or row.spo2 < min_spo2:
                    continue
                start = i * FPS - half
                stop = start + win
                if start < 0 or stop > n_frames:
                    continue
                seg = wave[:, start:stop]
                r.append(seg[0]); g.append(seg[1]); b.append(seg[2])
                spo2.append(row.spo2); pulse.append(row.pulse)
                subj.append(s); hand.append(h); idx.append(i)
    return {
        "r": np.array(r), "g": np.array(g), "b": np.array(b),
        "spo2": np.array(spo2), "pulse": np.array(pulse),
        "subject": np.array(subj), "hand": np.array(hand), "gt_index": np.array(idx),
    }


if __name__ == "__main__":
    print(alignment_report().to_string(index=False))
    d = build_windows()
    print(f"\n3-s windows: {len(d['spo2'])}")
