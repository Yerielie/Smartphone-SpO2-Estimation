# Smartphone Camera Oximetry: Paper Analysis and Reimplementation

## Contents
1. [Paper Analysis](#1-paper-analysis)
2. [Ratio-of-Ratios (Formula)](results/part1_summary.md)
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
