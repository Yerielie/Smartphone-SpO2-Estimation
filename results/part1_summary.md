# Part 1: Ratio-of-ratios on the Hoffman et al. (2022) dataset

All numbers below were produced by `python src/ratio_of_ratios.py`. Full-precision values are in
`part1_metrics.csv`, `part1_fold_coefficients.csv`, `part1_r_correlation.csv` and
`part1_alignment.csv`. This summary shows them to 2–3 decimals.

## Method

**Data.** 6 subjects × 2 hands (Nexus 6P, 30 fps), with ground truth from the Masimo Radical-7
(`SpO2 5`, `Pulse 5`) at 1 Hz. Ground-truth row *i* is aligned by index to frame *i*·30. Recordings
have 29.97–30.09 frames per ground-truth row, so the last row is off by between −0.97 s and +2.63 s
(`part1_alignment.csv`). This was reported and not corrected. Rows with missing SpO2 or SpO2 < 70
were excluded, following the paper, as were windows that run past either end of a recording.
This leaves **11,572 three-second windows**. Their mean ground-truth SpO2 is 88.43%, and 49.4% of
them are below 90%.

**Signal processing** (same as `src/spo2_hr_from_video.py`):
- 3-s window (90 frames) centred on frame *i*·30
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
| red/blue, fixed 110−25R | 8.18 ± 3.60 | 9.90 ± 4.89 | −0.72 ± 1.59 | 8.16 | 10.85 | −0.81 | −1.90 | 0.887 | 0.197 |
| **red/blue, calibrated LOSO** | **7.50 ± 1.99** | 8.91 ± 2.39 | −0.27 ± 0.48 | 7.45 | 9.13 | −0.29 | −0.62 | 0.947 | 0.052 |
| red/green, fixed 110−25R | 8.30 ± 2.81 | 10.69 ± 3.29 | −0.84 ± 0.78 | 8.15 | 10.94 | −0.85 | 5.07 | 0.132 | 0.831 |
| red/green, calibrated LOSO | 7.56 ± 2.25 | 9.12 ± 3.03 | −0.35 ± 0.73 | 7.51 | 9.49 | −0.39 | −0.67 | 0.961 | 0.049 |

By mean fold MAE, the best calibrated model is **red/blue** (`part1_bland_altman_best.png`). Its
Bland-Altman bias is −0.62, and its 95% limits of agreement are −18.48 to +17.24 percentage points.

### MAE per test subject

| Subject | n windows | red/blue fixed | red/blue calibrated | red/green fixed | red/green calibrated |
|---|---|---|---|---|---|
| 100001 | 1937 | 15.50 | 11.23 | 12.95 | 11.89 |
| 100002 | 2238 | 6.33 | 6.92 | 5.03 | 6.72 |
| 100003 | 2128 | 6.84 | 6.57 | 8.01 | 6.56 |
| 100004 | 2025 | 6.46 | 5.35 | 5.97 | 5.40 |
| 100005 | 1734 | 6.86 | 7.40 | 8.21 | 7.32 |
| 100006 | 1510 | 7.09 | 7.51 | 9.63 | 7.46 |

### R² per test subject

| Subject | red/blue fixed | red/blue calibrated | red/green fixed | red/green calibrated |
|---|---|---|---|---|
| 100001 | −3.95 | −1.25 | −2.14 | −1.83 |
| 100002 | −0.01 | −0.21 | 0.29 | −0.11 |
| 100003 | −0.04 | −0.04 | −0.86 | −0.04 |
| 100004 | −0.43 | −0.01 | −0.78 | −0.03 |
| 100005 | 0.08 | −0.04 | −0.62 | −0.03 |
| 100006 | 0.06 | −0.06 | −0.96 | −0.05 |

### Best model (red/blue, calibrated) per subject

| Subject | MAE | RMSE | R² | Bias | Sens <90 | Spec <90 |
|---|---|---|---|---|---|---|
| 100001 | 11.23 | 13.37 | −1.25 | −4.29 | 0.61 | 0.26 |
| 100002 | 6.92 | 8.15 | −0.21 | −2.03 | 1.00 | 0.00 |
| 100003 | 6.57 | 8.12 | −0.04 | 1.47 | 1.00 | 0.00 |
| 100004 | 5.35 | 6.18 | −0.01 | −1.11 | 1.00 | 0.00 |
| 100005 | 7.40 | 8.78 | −0.04 | 1.07 | 1.00 | 0.00 |
| 100006 | 7.51 | 8.89 | −0.06 | 1.94 | 1.00 | 0.00 |

## Comparison with the paper

| | MAE (%) | R² |
|---|---|---|
| Paper, ratio-of-ratios baseline (Hoffman et al. 2022) | 7.12 | 0.21 |
| This work, red/blue calibrated LOSO (folds mean) | 7.50 | −0.27 |
| This work, red/blue calibrated LOSO (pooled) | 7.45 | −0.29 |
| This work, red/blue fixed 110−25R (pooled) | 8.16 | −0.81 |

The MAE is close to the paper's baseline, but the R² is not. Our calibrated models have negative
R², so they do worse than predicting each subject's mean SpO2. The paper does not give every detail
of its baseline: the channel pair, how calibration was done, the validation split (the CNN used
4 train / 1 val / 1 test), any signal-quality filtering, and how R² was computed (pooled or per
subject). Its implementation may therefore differ from this one, and the two sets of numbers are
not strictly comparable.

## Why calibration does not help much

The main issue is that the linear fit across subjects comes out almost flat:

- **Coefficients from all subjects** (`part1_coefficients.json`): red/blue a = 85.54, b = −3.08;
  red/green a = 89.13, b = 1.06. For red/blue, b is negative, meaning predicted SpO2 *rises* with R,
  which is the opposite of the physiology. The JSON file flags this with a warning. Both slopes are
  so small that the model mostly predicts close to the training mean (about 88%).
- **Within each recording, R often does track SpO2 in the expected direction** (`part1_r_correlation.csv`).
  For red/green, Pearson r(R, SpO2) is negative in 9 of 12 recordings, down to −0.89 (100006 Left).
  Across all recordings pooled, though, r is −0.03 for red/green and +0.10 for red/blue.
- **R's offset differs between recordings.** The median red/blue R ranges from 0.75 to 1.39 across
  recordings, and subject 100001 Right (median 1.39) has a *positive* correlation. When 100001 is
  left out of training, the fitted slope has the expected sign (red/blue b = 12.21, red/green
  b = 19.60). In the other five folds, which include 100001 in training, the slope is near zero or
  inverted (`part1_fold_coefficients.csv`). Subject 100001 also has the largest error in every model.
- **The hypoxemia classifier is therefore close to trivial.** The calibrated models predict values
  near 88%, below the 90% cut-off, so almost every window is classified as hypoxemic. That gives a
  high sensitivity (0.95–0.96) but a specificity of about 0.05. In five of the six test subjects,
  every window is classified positive.

A single global linear calibration of the ratio-of-ratios cannot absorb the differences between
subjects and between hands. A per-subject or per-recording calibration, or signal-quality
filtering, might help, but those were outside this task and were not tested.

## Heart rate

The estimates are peak-based HR from the red channel, compared with Masimo `Pulse 5`. Only windows
that exist for both lengths and have a ground-truth pulse are compared (11,500 ten-second windows
are available).

| Window | Scope | n | Coverage (≥2 peaks) | MAE (bpm) | RMSE (bpm) | Bias (bpm) | Pearson r |
|---|---|---|---|---|---|---|---|
| 3 s | own valid windows | 10,942 | 95.1% | 4.83 | 11.64 | 0.99 | 0.669 |
| 10 s | own valid windows | 11,440 | 99.5% | 3.95 | 8.06 | −1.84 | 0.814 |
| 3 s | both valid | 10,931 | — | 4.79 | 11.48 | 0.95 | 0.675 |
| 10 s | both valid | 10,931 | — | 3.12 | 5.95 | −1.01 | 0.880 |

**A 10-second window helps.** On the same windows, MAE drops from 4.79 to 3.12 bpm and r rises from
0.675 to 0.880. Coverage also rises from 95.1% to 99.5%. The 3-s window produces outliers up to
about 225 bpm, when only two close peaks are found (`part1_hr_scatter.png`). The 10-s window mostly
under-estimates instead, with a cluster of low estimates for ground-truth pulse of 55–70 bpm.

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
