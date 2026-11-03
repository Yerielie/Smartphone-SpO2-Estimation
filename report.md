# Smartphone Camera Oximetry: Paper Analysis and Reimplementation

## Contents
1. [Paper Analysis](#1-paper-analysis)
2. [Ratio-of-Ratios (Formula)](#2-ratio-of-ratios-formula)
3. [CNN Reimplementation](#3-cnn-reimplementation)
4. Application to My Fingertip Video *(in progress)*
5. Discussion and Limitations *(in progress)*

---

## 1. Paper Analysis

**Hoffman, J. S., et al. (2022). Smartphone camera oximetry in an induced hypoxemia study. *npj Digital Medicine*, 5, 146.**

### 1.1 Background
Pulse oximeters estimate SpO2 from photoplethysmography (PPG), but they require a dedicated device. Smartphone-based approaches use the phone's camera and flash instead. Earlier smartphone studies relied on breath-holding, which only lowers SpO2 to around 80–85%, so their accuracy in the clinically important low range (below 90%) was unknown.

### 1.2 Study Design
- **Participants:** 6 healthy adults (3 female, 3 male; 1 with dark skin, 5 with light skin)
- **Protocol:** Varied FiO2 — participants breathed oxygen-nitrogen mixtures with decreasing oxygen, lowering SpO2 in steps toward four target plateaus (>93%, 89–93%, 80–88%, 70–79%) over about 13–19 minutes
- **Devices:** A Google Nexus 6P on the index finger of each hand, recording at 30 fps with the flash on. Exposure was locked, auto white balance was disabled, and per-channel color gains were fixed (red 1x, green 3x, blue 18x)
- **Ground truth:** Masimo Radical-7 pulse oximeter, with additional reference oximeters on other fingers

### 1.3 Method
1. Average the R, G, B pixel values of each frame → three PPG signals
2. Cut 3-second windows (90 frames) around each ground-truth reading
3. Train a CNN (3 convolutional layers + 2 fully connected layers) to predict SpO2 from each window
4. Evaluate with leave-one-subject-out cross-validation: 4 subjects for training, 1 for validation, 1 for testing, repeated 6 times

### 1.4 Results
| Metric | CNN | Ratio-of-ratios (baseline) |
|---|---|---|
| MAE | 5.00% | 7.12% |
| R² | 0.61 | 0.21 |

- Classifying SpO2 < 90%: sensitivity 81%, specificity 79%
- RMS error (Arms) 5.55%, above the 3.5% required by the ISO 80601-2-61 standard for medical pulse oximeters

### 1.5 Contributions
- First clinical validation of an unmodified smartphone camera across SpO2 70–100%
- First to report smartphone camera accuracy below 85% SpO2
- Released the dataset publicly (MIT license)

### 1.6 Limitations
- **Small sample:** 6 subjects; ISO standards require at least 10 with diverse skin tones
- **Individual differences:** One subject with fingertip calluses had the largest error (MAE 8.56%)
- **Skin tone and sex:** Too few subjects to draw conclusions
- **Low SpO2:** The model over-predicted below 80% for 3 subjects
- **Single device:** Only one phone model tested
- **Clinical use:** The authors state the accuracy is not sufficient for medical use

### 1.7 Implications for This Project
- The CNN code was not released, so the model is reimplemented from the paper's description (Section 3)
- The ratio-of-ratios baseline (MAE 7.12%) gives a reference point for the calibrated formula (Section 2)
- The single-device limitation is tested directly by applying the models to a video from a different phone (Section 4)

---

## 2. Ratio-of-Ratios (Formula)

### 2.1 Background
Ratio-of-ratios is the classical method used in medical pulse oximeters. It compares the pulsatile signal at two wavelengths to estimate SpO2 with a fixed formula. The paper used it as a baseline for its CNN (MAE 7.12%, R² 0.21) but did not publish the exact implementation. This section tests how well the method works on smartphone camera signals.

### 2.2 Data
- **Source:** The paper's public dataset (6 subjects × 2 hands)
- **Input:** Smartphone RGB signals, 30 fps
- **Ground truth:** Masimo Radical-7 SpO2 and pulse, 1 Hz
- **Exclusion:** SpO2 < 70%, following the paper
- **Samples:** 11,568 three-second windows

### 2.3 Method
1. Match each ground-truth reading to a 3-second window (90 frames) of the RGB signal
2. Calculate R = (AC_red / DC_red) / (AC_blue / DC_blue), where AC is the pulsatile part (band-pass 0.67–4.0 Hz) and DC is the mean signal level
3. Estimate SpO2 in two ways:
   - **Fixed formula:** SpO2 = 110 − 25R, designed for medical pulse oximeters
   - **Calibrated formula:** SpO2 = a − b × R, with a and b fitted to the smartphone data
4. Evaluate the calibrated formula with leave-one-subject-out cross-validation: 5 subjects for fitting, 1 for testing, repeated 6 times
5. Estimate heart rate by peak detection on the red signal and compare with the Masimo pulse, using 3-second and 10-second windows

### 2.4 Results
| Metric | Fixed formula | Calibrated formula | Paper (baseline) |
|---|---|---|---|
| MAE | 8.18% | 7.49% | 7.12% |
| R² | −0.71 | −0.26 | 0.21 |

- Mean across the 6 test subjects. The red/green channel pair gave similar results.
- Classifying SpO2 < 90% (calibrated formula): sensitivity 94%, specificity 5%
- Heart rate: MAE 3.22 bpm with a 10-second window, 4.68 bpm with a 3-second window

### 2.5 Findings
- **Calibration failed:** MAE is close to the paper's baseline, but R² is negative. The calibrated formula predicts about 88% for everyone instead of following SpO2 changes. Fitted on all subjects, the slope has the wrong sign (b = −3.15).
- **Individual differences:** Within one recording, R follows SpO2 (correlation down to −0.89), but the baseline level of R differs between subjects and hands. One subject (100001) reverses the overall trend; the paper also reported the largest error for this subject, who had fingertip calluses.
- **Low-SpO2 detection:** Because predictions stay near 88%, almost every sample is classified as below 90%.
- **Heart rate:** Peak detection works, and a 10-second window is more stable than a 3-second window.

### 2.6 Implications for Section 3
- A single formula cannot handle the differences between people, so a data-driven model (CNN) is needed
- Heart rate for new videos will use the 10-second window (Section 4)

Full results, per-subject tables and figures: [results/section2_summary.md](results/section2_summary.md)

---

## 3. CNN Reimplementation

### 3.1 Background
Section 2 showed that a single formula cannot handle the differences between people. The paper addressed this with a CNN that learns the relationship between the RGB signal and SpO2 from data (MAE 5.00%, R² 0.61). The authors did not release the model code, so this section reimplements the CNN from the paper's description and evaluates it on the same public dataset.

### 3.2 Data
- **Source:** The paper's public dataset (6 subjects × 2 hands), same as Section 2
- **Input:** One 3-second window of the R, G, B signals (3 × 90 frames)
- **Ground truth:** Masimo Radical-7 SpO2, 1 Hz
- **Exclusion:** SpO2 < 70%, following the paper
- **Samples:** 11,568 three-second windows

### 3.3 Method
1. Build the CNN described in the paper: 3 convolutional layers + 2 fully connected layers, ReLU, kernel size 3, no padding
2. Train with the paper's settings: MSE loss, Adam (learning rate 0.00001, reduced 10-fold after 80 epochs), L2 regularisation 0.1
3. Evaluate with leave-one-subject-out cross-validation: 4 subjects for training, 1 for validation, 1 for testing, repeated 6 times
4. Repeat the whole evaluation with 3 random seeds and report the mean ± SD, so the result does not depend on one run
5. Train a final model on all 6 subjects for use on new videos (Section 4)

Settings not given in the paper (number of filters, input normalisation, batch size, number of epochs) were fixed before training and are listed in the full results. No setting was tuned on the test subjects.

### 3.4 Results
| Metric | Paper CNN | Section 2 (calibrated formula) | Section 3 CNN |
|---|---|---|---|
| MAE | 5.00% | 7.49% | 5.24 ± 0.77% |
| R² | 0.61 | −0.26 | 0.34 ± 0.18 |
| Sensitivity (SpO2 < 90%) | 81% | 94% | 87 ± 6% |
| Specificity (SpO2 < 90%) | 79% | 5% | 61 ± 11% |
| Arms | 5.55% | — | 6.49 ± 0.95% |

- MAE and R²: mean across the 6 test subjects. Sensitivity, specificity and Arms: all test samples pooled. ± is the SD across 3 seeds.
- MAE by seed: 4.73%, 6.13%, 4.84%

### 3.5 Findings
- **Improvement over the formula:** MAE fell from 7.49% to 5.24% and R² became positive. The CNN had a lower error than the formula for every subject, and R² was positive for 5 of 6 subjects (0 of 6 in Section 2).
- **Gap with the paper:** MAE is close to the paper's, but R² and specificity are lower. Predictions are pulled toward the average: below 80% they are over-predicted by 7–10 points, and above 95% under-predicted by 4–6 points. The paper also reported over-prediction below 80%.
- **Variation between runs:** Results changed noticeably with the random seed. The stopping point of training is chosen with a single validation subject, and this choice was unstable (best epoch ranged from 2 to 77).
- **Individual differences:** Subject 100001 again had the largest error (MAE 7.55%), but less than in Section 2 (11.17%) and the paper (8.56%).
- **Clinical use:** Arms is above the 3.5% required by ISO 80601-2-61, as in the paper.

### 3.6 Implications for Section 4
- The final model was trained on all 6 subjects for 16 epochs (the median best epoch from cross-validation)
- The training data was recorded on one phone (Nexus 6P) with locked camera settings, so a video from a different phone is likely to give larger errors
- Because results vary between runs, averaging several models may give more stable estimates on new videos
- Heart rate will use the 10-second window from Section 2

Full results, per-subject tables and figures: [results/section3_summary.md](results/section3_summary.md)
