# Part 1: Ratio-of-ratios on the Hoffman et al. (2022) dataset

All numbers below were produced by `python src/ratio_of_ratios.py`. Full-precision values are in
`part1_metrics.csv`, `part1_fold_coefficients.csv`, `part1_r_correlation.csv` and
`part1_alignment.csv`. This summary shows them to 2–3 decimals.

## Method

**Data.** 6 subjects × 2 hands (Nexus 6P, 30 fps), with ground truth from the Masimo Radical-7
(`SpO2 5`, `Pulse 5`) at 1 Hz. Rows with missing SpO2 or SpO2 < 70 were excluded, following the
paper, as were windows that run past either end of a recording. This leaves **11,568 three-second
windows**. Their mean ground-truth SpO2 is 88.43%, and 49.4% of them are below 90%.

**Time alignment.** The camera and oximeters were started and stopped together, but the
recordings have 29.97–30.09 frames per ground-truth row rather than exactly 30
(`part1_alignment.csv`). Each recording is therefore stretched to its own length: ground-truth
row *i* maps to frame round(*i* · n_frames / n_gt_rows), so the first and last rows line up with
both ends of the video. The earlier version used frame *i* · 30, which drifted by between −0.97 s
and +2.63 s by the end of a recording. Both versions are compared [below](#before--after-the-alignment-fix).

**Signal processing** (same as `src/spo2_hr_from_video.py`):
- 3-s window (90 frames) centred on the aligned frame
- Butterworth band-pass, order 1, 0.67–4.0 Hz, `filtfilt`
- AC = std of the filtered window, DC = mean of the raw window
- R = (AC_red/DC_red) / (AC_x/DC_x), where x = blue or green
- Heart rate: `find_peaks` on the filtered red signal (height = std, distance = fs/4), HR = 60 / mean peak interval

**Models.**
- *Fixed:* SpO2 = 110 − 25R. No calibration and no clamping.
- *Calibrated:* least-squares fit of SpO2 = a − b·R with leave-one-subject-out CV. There are 6 folds;
  each fits on 5 subjects and tests on 1, with both hands always in the same fold.

**Metrics.** MAE, RMSE, R² and bias (prediction − ground truth) are computed per test subject.
"Folds" gives the mean ± SD over the 6 subjects; "pooled" uses all test predictions together.
For hypoxemia classification, positive means SpO2 < 90%, applied to both ground truth and prediction.

## SpO2 results

| Model | MAE (folds) | RMSE (folds) | R² (folds) | MAE (pooled) | RMSE (pooled) | R² (pooled) | Bias (pooled) | Sens <90 (pooled) | Spec <90 (pooled) |
|---|---|---|---|---|---|---|---|---|---|
| red/blue, fixed 110−25R | 8.18 ± 3.59 | 9.89 ± 4.85 | −0.71 ± 1.57 | 8.16 | 10.82 | −0.81 | −1.87 | 0.883 | 0.196 |
| **red/blue, calibrated LOSO** | **7.49 ± 1.97** | 8.90 ± 2.36 | −0.26 ± 0.47 | 7.44 | 9.12 | −0.28 | −0.61 | 0.944 | 0.051 |
| red/green, fixed 110−25R | 8.31 ± 2.81 | 10.70 ± 3.29 | −0.85 ± 0.78 | 8.15 | 10.96 | −0.85 | 5.09 | 0.129 | 0.831 |
| red/green, calibrated LOSO | 7.55 ± 2.23 | 9.12 ± 3.01 | −0.35 ± 0.72 | 7.50 | 9.47 | −0.38 | −0.65 | 0.963 | 0.050 |

By mean fold MAE, the best calibrated model is **red/blue** (`part1_bland_altman_best.png`). Its
Bland-Altman bias is −0.61, and its 95% limits of agreement are −18.43 to +17.22 percentage points.

### MAE per test subject

| Subject | n windows | red/blue fixed | red/blue calibrated | red/green fixed | red/green calibrated |
|---|---|---|---|---|---|
| 100001 | 1936 | 15.48 | 11.17 | 12.95 | 11.85 |
| 100002 | 2238 | 6.34 | 6.93 | 5.04 | 6.72 |
| 100003 | 2126 | 6.84 | 6.57 | 8.03 | 6.56 |
| 100004 | 2024 | 6.47 | 5.34 | 5.98 | 5.40 |
| 100005 | 1734 | 6.85 | 7.40 | 8.21 | 7.32 |
| 100006 | 1510 | 7.09 | 7.51 | 9.64 | 7.46 |

### R² per test subject

| Subject | red/blue fixed | red/blue calibrated | red/green fixed | red/green calibrated |
|---|---|---|---|---|
| 100001 | −3.90 | −1.22 | −2.14 | −1.81 |
| 100002 | −0.01 | −0.21 | 0.29 | −0.11 |
| 100003 | −0.05 | −0.04 | −0.87 | −0.04 |
| 100004 | −0.43 | −0.01 | −0.78 | −0.03 |
| 100005 | 0.08 | −0.04 | −0.62 | −0.03 |
| 100006 | 0.06 | −0.06 | −0.96 | −0.05 |

### Best model (red/blue, calibrated) per subject

| Subject | MAE | RMSE | R² | Bias | Sens <90 | Spec <90 |
|---|---|---|---|---|---|---|
| 100001 | 11.17 | 13.29 | −1.22 | −4.21 | 0.59 | 0.26 |
| 100002 | 6.93 | 8.16 | −0.21 | −2.03 | 1.00 | 0.00 |
| 100003 | 6.57 | 8.11 | −0.04 | 1.48 | 1.00 | 0.00 |
| 100004 | 5.34 | 6.18 | −0.01 | −1.11 | 1.00 | 0.00 |
| 100005 | 7.40 | 8.78 | −0.04 | 1.06 | 1.00 | 0.00 |
| 100006 | 7.51 | 8.89 | −0.06 | 1.93 | 1.00 | 0.00 |

## Comparison with the paper

| | MAE (%) | R² |
|---|---|---|
| Paper, ratio-of-ratios baseline (Hoffman et al. 2022) | 7.12 | 0.21 |
| This work, red/blue calibrated LOSO (folds mean) | 7.49 | −0.26 |
| This work, red/blue calibrated LOSO (pooled) | 7.44 | −0.28 |
| This work, red/blue fixed 110−25R (pooled) | 8.16 | −0.81 |

The MAE is close to the paper's baseline, but the R² is not. Our calibrated models have negative
R², so they do worse than predicting each subject's mean SpO2. The paper does not give every detail
of its baseline: the channel pair, how calibration was done, the validation split (the CNN used
4 train / 1 val / 1 test), any signal-quality filtering, and how R² was computed (pooled or per
subject). Its implementation may therefore differ from this one, and the two sets of numbers are
not strictly comparable.

## Why calibration does not help much

The main issue is that the linear fit across subjects comes out almost flat:

- **Coefficients from all subjects** (`part1_coefficients.json`): red/blue a = 85.48, b = −3.15;
  red/green a = 89.09, b = 1.01. For red/blue, b is negative, meaning predicted SpO2 *rises* with R,
  which is the opposite of the physiology. The JSON file flags this with a warning. Both slopes are
  so small that the model mostly predicts close to the training mean (about 88%).
- **Within each recording, R often does track SpO2 in the expected direction** (`part1_r_correlation.csv`).
  For red/green, Pearson r(R, SpO2) is negative in 9 of 12 recordings, down to −0.89 (100006 Left).
  Across all recordings pooled, though, r is −0.03 for red/green and +0.10 for red/blue.
- **R's offset differs between recordings.** The median red/blue R ranges from 0.74 to 1.39 across
  recordings, and subject 100001 Right (median 1.39) has a *positive* correlation. When 100001 is
  left out of training, the fitted slope has the expected sign (red/blue b = 12.11, red/green
  b = 19.47). In the other five folds, which include 100001 in training, the slope is near zero or
  inverted (`part1_fold_coefficients.csv`). Subject 100001 also has the largest error in every model.
- **The hypoxemia classifier is therefore close to trivial.** The calibrated models predict values
  near 88%, below the 90% cut-off, so almost every window is classified as hypoxemic. That gives a
  high sensitivity (0.94–0.96) but a specificity of about 0.05. In five of the six test subjects,
  every window is classified positive.

A single global linear calibration of the ratio-of-ratios cannot absorb the differences between
subjects and between hands. A per-subject or per-recording calibration, or signal-quality
filtering, might help, but those were outside this task and were not tested.

## Heart rate

The estimates are peak-based HR from the red channel, compared with Masimo `Pulse 5`. Only windows
that exist for both lengths and have a ground-truth pulse are compared (11,496 ten-second windows
are available).

| Window | Scope | n | Coverage (≥2 peaks) | MAE (bpm) | RMSE (bpm) | Bias (bpm) | Pearson r |
|---|---|---|---|---|---|---|---|
| 3 s | own valid windows | 10,981 | 95.5% | 4.72 | 10.76 | 0.75 | 0.699 |
| 10 s | own valid windows | 11,439 | 99.5% | 4.00 | 8.35 | −1.83 | 0.803 |
| 3 s | both valid | 10,969 | — | 4.68 | 10.60 | 0.72 | 0.705 |
| 10 s | both valid | 10,969 | — | 3.22 | 6.34 | −1.08 | 0.867 |

**A 10-second window helps.** On the same windows, MAE drops from 4.68 to 3.22 bpm and r rises from
0.705 to 0.867. Coverage also rises from 95.5% to 99.5%. The 3-s window produces outliers up to
225 bpm, when only two close peaks are found (`part1_hr_scatter.png`). The 10-s window mostly
under-estimates instead, with a cluster of low estimates for ground-truth pulse of 55–70 bpm.

## Before / after the alignment fix

The table compares the earlier alignment (row *i* → frame *i* · 30) with the current one
(row *i* → frame round(*i* · n_frames / n_gt_rows)). The rest of the pipeline is unchanged.
The window count drops from 11,572 to 11,568 three-second windows and from 11,500 to 11,496
ten-second windows. Only the last one or two ground-truth rows of each recording are affected:
depending on whether a recording has more or fewer than 30 frames per row, a window there now
does or no longer fits inside the video. In total this is a net loss of 4 windows.

| Model | Scope | MAE | RMSE | R² | Sens <90 | Spec <90 |
|---|---|---|---|---|---|---|
| red/blue, fixed | folds mean | 8.18 → 8.18 | 9.90 → 9.89 | −0.72 → −0.71 | 0.875 → 0.871 | 0.188 → 0.186 |
| red/blue, fixed | pooled | 8.16 → 8.16 | 10.85 → 10.82 | −0.81 → −0.81 | 0.887 → 0.883 | 0.197 → 0.196 |
| red/blue, calibrated | folds mean | 7.50 → 7.49 | 8.91 → 8.90 | −0.27 → −0.26 | 0.935 → 0.931 | 0.043 → 0.043 |
| red/blue, calibrated | pooled | 7.45 → 7.44 | 9.13 → 9.12 | −0.29 → −0.28 | 0.947 → 0.944 | 0.052 → 0.051 |
| red/green, fixed | folds mean | 8.30 → 8.31 | 10.69 → 10.70 | −0.84 → −0.85 | 0.139 → 0.136 | 0.863 → 0.863 |
| red/green, fixed | pooled | 8.15 → 8.15 | 10.94 → 10.96 | −0.85 → −0.85 | 0.132 → 0.129 | 0.831 → 0.831 |
| red/green, calibrated | folds mean | 7.56 → 7.55 | 9.12 → 9.12 | −0.35 → −0.35 | 0.951 → 0.954 | 0.041 → 0.042 |
| red/green, calibrated | pooled | 7.51 → 7.50 | 9.49 → 9.47 | −0.39 → −0.38 | 0.961 → 0.963 | 0.049 → 0.050 |

| Heart rate | Scope | n | MAE (bpm) | RMSE (bpm) | Pearson r | Coverage |
|---|---|---|---|---|---|---|
| 3-s window | own valid windows | 10,942 → 10,981 | 4.83 → 4.72 | 11.64 → 10.76 | 0.669 → 0.699 | 0.951 → 0.955 |
| 10-s window | own valid windows | 11,440 → 11,439 | 3.95 → 4.00 | 8.06 → 8.35 | 0.814 → 0.803 | 0.995 → 0.995 |
| 3-s window | both valid | 10,931 → 10,969 | 4.79 → 4.68 | 11.48 → 10.60 | 0.675 → 0.705 | — |
| 10-s window | both valid | 10,931 → 10,969 | 3.12 → 3.22 | 5.95 → 6.34 | 0.880 → 0.867 | — |

**What changed:**
- **SpO2 metrics barely move.** In the fold-mean and pooled rows above, MAE changes by at most
  0.01, RMSE by 0.03 and R² by 0.01. Per test subject, the largest changes are 0.06 in MAE, 0.10 in
  RMSE and 0.05 in R² (all for subject 100001). None of the conclusions change. SpO2 changes slowly compared with a drift of a few seconds, so the exact
  window position matters little for a 1 Hz ground truth.
- **3-s heart rate improves slightly** (pooled MAE 4.83 → 4.72 bpm, r 0.669 → 0.699).
- **10-s heart rate gets slightly worse** (pooled MAE 3.95 → 4.00 bpm, r 0.814 → 0.803). The
  10-s window is still clearly better than the 3-s window.
- **The calibration coefficients shift a little** (red/blue a 85.54 → 85.48, b −3.08 → −3.15;
  red/green a 89.13 → 89.09, b 1.06 → 1.01). red/blue's b is still negative.

## Output files

| File | Content |
|---|---|
| `part1_metrics.csv` | Every metric per model and scope (per subject, folds mean, folds sd, pooled), plus HR |
| `part1_coefficients.json` | a, b fitted on all subjects for both pairs, plus preprocessing details |
| `part1_fold_coefficients.csv` | a, b for each LOSO fold |
| `part1_r_correlation.csv` | Median R and r(R, SpO2) for each recording |
| `part1_alignment.csv` | Frame count vs ground-truth length for each recording |
| `part1_scatter_red_blue.png`, `part1_scatter_red_green.png` | Predicted vs ground-truth SpO2 |
| `part1_bland_altman_best.png` | Bland-Altman plot for red/blue calibrated |
| `part1_hr_scatter.png` | Estimated HR vs ground truth, 3-s and 10-s windows |
