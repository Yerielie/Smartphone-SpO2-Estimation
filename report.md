# Smartphone Camera Oximetry: Paper Analysis and Reimplementation

## Contents
1. [Paper Analysis](#1-paper-analysis)
2. [Ratio-of-Ratios (Formula)](#2-ratio-of-ratios-formula)
3. CNN Reimplementation *(in progress)*
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
- Classifying SpO2 < 90% (calibrated formula): sensitivity 0.94, specificity 0.05
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
