# Section 3: CNN on the Hoffman et al. (2022) dataset

All numbers below were produced by `python src/public_data_spo2_cnn.py`. Full-precision values are in
`section3_metrics.csv`, `section3_per_subject.csv`, `section3_folds.csv` and `section3_loss_history.csv`.
This summary rounds them to 2 decimals. A full run (18 LOSO models and the final model) takes about
20 minutes on 4 CPU cores.

## Method

**Data.** The data is the same as in Section 2. `src/dataset.py` (`build_windows`) is reused unchanged: the same
per-recording time alignment, the same exclusion of missing or < 70% SpO2, and 3-s windows (90 frames)
centred on each ground-truth second. This gives **11,568 windows**, the same count as Section 2. Each
window is one input of shape 3 × 90 (R, G, B frame means, raw values) with one target, the Masimo
Radical-7 SpO2 (`SpO2 5`).

**Model** (`SpO2CNN`, 44,337 parameters):

```
input (3, 90)
Conv1d(3→8,  k=3, no padding) + ReLU   → (8, 88)
Conv1d(8→16, k=3, no padding) + ReLU   → (16, 86)
Conv1d(16→16, k=3, no padding) + ReLU  → (16, 84)
Flatten → Linear(1344→32) + ReLU → Linear(32→1)   = SpO2
```

**Training.** MSE loss. Adam with lr 1e-5, multiplied by 0.1 after epoch 80, and L2 weight decay 0.1.
PyTorch, CPU.

**Evaluation.** Leave-one-subject-out over 6 folds. Each fold has 1 test subject, 1 validation subject and
4 training subjects, and both hands of a subject are always in the same split. The validation subject is
used only to pick the epoch. The normalisation is fitted on the 4 training subjects only. Each fold is
trained with 3 seeds (0, 1, 2), which use the same splits but different weight initialisation and batch
order. The metrics are the same functions as in Section 2 (`all_metrics`):
- **Per test subject**, then mean ± SD over the 6 subjects ("folds").
- **Pooled** over all test predictions.

Each of these is computed separately for each seed. The tables report the **mean ± SD across the 3 seeds**.

### What comes from the paper and what is assumed

Taken from the paper's description (as given in the task):
- 3-s RGB window input
- 3 convolutional + 2 fully connected layers, ReLU, kernel size 3, no padding
- SpO2 regression output
- MSE loss
- Adam, lr 1e-5, decayed by 0.1 after 80 epochs
- L2 regularisation 0.1
- PyTorch
- 4 / 1 / 1 LOSO split

**Assumptions (not specified by the paper; chosen here):**

| Item | Choice | Why |
|---|---|---|
| Convolution type | 1D over time, R/G/B as 3 input channels | The input is 3 time series; a 2D kernel of size 3 over a 3-row "image" would collapse the channel axis after one layer |
| Filters | 8, 16, 16 | Keep the model small (about 44k parameters for about 7–8k training windows per fold) |
| Pooling / dropout / batch norm | None | Not mentioned; kept minimal |
| Hidden FC size | 32 | Small |
| Input normalisation | Per-channel z-score using the mean / SD of all training-subject frames | Simple and keeps the DC level. Fitted on the training subjects only |
| Signal filtering | None (raw frame means) | Lets the CNN learn its own filters |
| Target normalisation | SpO2 z-scored with the training mean / SD; predictions mapped back to % | With lr 1e-5, regressing raw values around 88 would need very many epochs just to learn the offset |
| L2 implementation | `torch.optim.Adam(weight_decay=0.1)` (L2 added to the gradient, not AdamW) | Closest reading of "L2 regularisation" with Adam |
| Batch size | 64 | Common default |
| Epochs | 100 (so the decay at 80 takes effect) | Not specified |
| Early stopping / model selection | Train all 100 epochs and keep the weights from the epoch with the lowest validation MSE | Uses the validation subject only |
| Validation subject | The next subject after the test subject, cyclic (test 100001 → val 100002, …, test 100006 → val 100001) | Deterministic; the paper does not say how it was chosen |
| Seeds | 0, 1, 2 | As requested |
| Output clamping | None | Same as Section 2; 1.2–1.6% of predictions are above 100% |
| Arms | sqrt(mean((pred − ref)²)), the ISO 80601-2-61 definition | Identical to RMSE against the reference |

Nothing was tuned on the test subjects. All the settings above were fixed before the first run, and this
summary reports that single run.

## Results

### Summary (mean ± SD across 3 seeds)

| Scope | MAE | RMSE / Arms | R² | Bias | Sens <90 | Spec <90 |
|---|---|---|---|---|---|---|
| Folds mean (mean of 6 test subjects) | **5.24 ± 0.77** | 6.34 ± 0.88 | **0.34 ± 0.18** | 0.16 ± 0.40 | 0.86 ± 0.06 | 0.63 ± 0.11 |
| Folds SD (spread between subjects) | 1.47 ± 0.73 | 1.72 ± 0.66 | 0.28 ± 0.13 | 2.73 ± 1.17 | 0.14 ± 0.02 | 0.24 ± 0.04 |
| Pooled (all test windows) | 5.22 ± 0.72 | 6.49 ± 0.95 | 0.34 ± 0.20 | 0.00 ± 0.39 | **0.87 ± 0.06** | **0.61 ± 0.11** |

**Per seed (folds mean / pooled).** The spread between seeds is large:

| Seed | MAE folds | R² folds | MAE pooled | R² pooled | Arms pooled | Sens pooled | Spec pooled |
|---|---|---|---|---|---|---|---|
| 0 | 4.73 | 0.46 | 4.74 | 0.48 | 5.79 | 0.80 | 0.72 |
| 1 | 6.13 | 0.14 | 6.04 | 0.12 | 7.57 | 0.91 | 0.50 |
| 2 | 4.84 | 0.42 | 4.87 | 0.42 | 6.11 | 0.89 | 0.61 |

The plots show seed 0. Seed 0 was chosen as the plotting seed before training, but it turned out to be
the best of the three. The scatter and Bland-Altman plots therefore look somewhat better than the
seed average in the tables.

### Per test subject (mean ± SD across 3 seeds)

| Subject | n windows | MAE | RMSE | R² | Bias | Sens <90 | Spec <90 | Section 2 calibrated MAE |
|---|---|---|---|---|---|---|---|---|
| 100001 | 1936 | **7.55 ± 2.54** | 8.69 ± 2.65 | −0.01 ± 0.62 | −2.53 ± 4.46 | 0.90 ± 0.09 | 0.48 ± 0.41 | 11.17 |
| 100002 | 2238 | 4.20 ± 0.49 | 4.97 ± 0.55 | 0.55 ± 0.10 | −0.85 ± 1.16 | 0.93 ± 0.06 | 0.67 ± 0.11 | 6.93 |
| 100003 | 2126 | 4.99 ± 0.14 | 6.02 ± 0.11 | 0.43 ± 0.02 | −0.83 ± 2.39 | 0.98 ± 0.02 | 0.38 ± 0.06 | 6.57 |
| 100004 | 2024 | 4.37 ± 0.27 | 5.04 ± 0.19 | 0.33 ± 0.05 | −0.15 ± 0.39 | 0.88 ± 0.08 | 0.63 ± 0.08 | 5.34 |
| 100005 | 1734 | 5.81 ± 0.18 | 7.68 ± 0.21 | 0.21 ± 0.04 | 3.68 ± 0.59 | 0.62 ± 0.08 | 0.81 ± 0.03 | 7.40 |
| 100006 | 1510 | 4.49 ± 1.83 | 5.63 ± 2.07 | 0.54 ± 0.35 | 1.66 ± 0.41 | 0.88 ± 0.12 | 0.84 ± 0.28 | 7.51 |

The CNN has a lower MAE than the Section 2 calibrated formula for every subject. R² is positive for
5 of 6 subjects; in Section 2 it was negative for all 6.

### Comparison with the paper

| | MAE | R² |
|---|---|---|
| Paper CNN | 5.00% | 0.61 |
| Paper ratio-of-ratios | 7.12% | 0.21 |
| This repo, Section 2 calibrated formula | 7.49% | −0.26 |
| This repo, Section 3 CNN (folds mean, mean ± SD over 3 seeds) | **5.24 ± 0.77%** | **0.34 ± 0.18** |
| This repo, Section 3 CNN (pooled, mean ± SD over 3 seeds) | 5.22 ± 0.72% | 0.34 ± 0.20 |

| SpO2 < 90% | Sensitivity | Specificity |
|---|---|---|
| Paper CNN | 81% | 79% |
| This repo, Section 2 calibrated formula (pooled) | 94% | 5% |
| This repo, Section 3 CNN (pooled, mean ± SD over 3 seeds) | 87 ± 6% | 61 ± 11% |

| Arms | |
|---|---|
| Paper CNN | 5.55% |
| This repo, Section 3 CNN (pooled) | 6.49 ± 0.95% |

**Where we stand.**
- **MAE is close to the paper's** (5.24 vs 5.00), and well inside the seed-to-seed SD.
- **R² is clearly worse** (0.34 vs 0.61).
- **Specificity for SpO2 < 90% is lower** (61% vs 79%): too many normoxic windows are called hypoxemic.
  Sensitivity is somewhat higher (87% vs 81%).
- **Arms is higher** (6.49 vs 5.55). Both are above the 3.5% that ISO 80601-2-61 requires.

**Compared with Section 2, the CNN is a large step forward.** MAE drops from 7.49 to 5.24 and R² rises
from −0.26 to 0.34. The hypoxemia classifier is no longer trivial: specificity rises from 5% to 61%.

### Why the R² gap? (likely reasons, not tested)

1. **Predictions are pulled toward the mean.** Pooled over test subjects (per seed), windows with
   ground truth < 80% are over-predicted by +7.1 / +10.4 / +9.5 points on average, and windows ≥ 95%
   are under-predicted by −4.1 / −6.4 / −3.9. The scatter plot shows the same thing: its slope is
   flatter than the identity line. A flattened slope costs a lot of R² while costing less MAE, which
   fits a near-paper MAE with a much lower R². The paper also reports over-prediction below 80% for
   3 subjects.
2. **The model underfits even the training subjects.** Training MSE plateaus at about 21–28 (%²), an
   RMSE of about 5%, in every fold (`section3_loss_curves.png`). With lr 1e-5 and a coupled L2 weight
   decay of 0.1 on a small network, the regularisation is strong. The paper may have used a larger
   network, or applied the 0.1 differently (for example, a loss term λ·Σw² with a different scaling,
   or AdamW-style decoupled decay). These details are not in the description, so they were not changed
   here.
3. **Model selection uses one validation subject, and that subject is noisy.** Validation loss reaches
   its minimum early: the median best epoch is 16, and 14 of the 18 runs pick an epoch ≤ 24.
   After that the validation curve flattens or rises while training loss keeps falling slowly. As a
   result, the lr decay at epoch 80 almost never affects the chosen model. How good the chosen epoch is
   depends a lot on who the validation subject is:
   - When 100001 is the validation subject (fold "test 100006"), the validation MSE ranges from 41 to
     200 depending on the seed. Seed 1 picked epoch 2 and got a test MAE of 6.59, against 3.66 and 3.23
     for the other seeds.
   - When 100005 is the validation subject, the validation curve is flat at about 55–60 after epoch 5,
     so it carries little information.

   The paper does not say how it chose the validation subject, so its selection may simply have been
   luckier or better matched.
4. **The input representation is a guess.** The paper does not describe its input normalisation. A
   global per-channel z-score keeps each recording's DC level. The DC level differs a lot between
   subjects and hands (Section 2 found the same for R), and the AC pulsation is small in comparison.
   Per-window normalisation or band-pass filtering might generalise differently. Comparing these
   choices would have to be done on the validation subjects only, and was not part of this task.
5. **Seed variance is large** (folds-mean MAE 4.73 / 6.13 / 4.84). With 6 subjects, the paper's single
   reported number could sit anywhere in a similar spread. The paper does not say whether it averaged
   over seeds.

### Subject 100001

Subject 100001 is again the hardest subject:
- **Section 2:** MAE 11.17 with the calibrated formula. This subject reversed the overall R–SpO2 trend.
- **Paper:** MAE 8.56, the largest of all subjects. The authors attribute this to fingertip calluses.
- **Here:** MAE 7.55 ± 2.54 and R² −0.01 ± 0.62, the largest error and the only subject with an R² of
  about 0.

The result is also very seed-dependent: MAE 5.57 / 10.42 / 6.67. Training loss in that fold looks
normal; what varies is how well the model transfers to this subject. Below 80% ground truth, seed 0
over-predicts 100001 by +10.2 points on average.

As a *validation* subject, 100001 also makes model selection unstable for test subject 100006 (see
point 3 above). Both observations fit the idea that this subject's signals differ from the rest, for
example because of calluses changing light transmission, so the other 5 subjects do not cover them well.

## Figures

| File | Content |
|---|---|
| `section3_scatter.png` | Predicted vs ground-truth SpO2, seed 0, all LOSO test windows |
| `section3_bland_altman.png` | Bland-Altman plot, seed 0 (bias −0.21; 95% limits of agreement −11.55 to +11.12) |
| `section3_loss_curves.png` | Train and validation MSE per epoch for each fold and seed, with the best epochs marked in each panel title |

## Final model for Section 4

`section3_model.pt` holds a model trained on **all 6 subjects** (11,568 windows), together with the
normalisation parameters (`x_mean`, `x_std`, `y_mean`, `y_std`), filter counts and window length.
`section3_model.json` describes the input format and preprocessing.
- **Training length.** No subject is left for early stopping, so the model is trained for 16 epochs.
  This is the median best-validation epoch over the 18 LOSO runs, so it was chosen on validation data
  only. Seed 0. Training MSE at the end is 28.45.
- **Not evaluated.** The final model has no held-out estimate of its own. Expect performance like the
  LOSO results above, and probably worse on a different phone (see below).
- **Input.** A (3, 90) array of raw R, G, B frame means at 30 fps. The model was trained on Nexus 6P
  recordings with locked exposure, no auto white balance and fixed channel gains (R 1×, G 3×, B 18×). A
  video from another phone will have different absolute channel levels. Because the input z-score keeps
  the DC level, this is a likely source of error in Section 4.

## Output files

| File | Content |
|---|---|
| `section3_metrics.csv` | Every metric per seed and scope (per subject, folds mean, folds sd, pooled), plus mean and sd over seeds |
| `section3_per_subject.csv` | Per-subject metrics, mean and sd over seeds |
| `section3_folds.csv` | Split sizes, best epoch and best validation MSE for each fold × seed |
| `section3_loss_history.csv` | Train and validation MSE per epoch for each fold × seed |
| `section3_scatter.png`, `section3_bland_altman.png`, `section3_loss_curves.png` | Figures |
| `section3_model.pt`, `section3_model.json` | Final model trained on all subjects, with its preprocessing |

Window-level predictions are not saved, because they would repeat the dataset's ground truth.
