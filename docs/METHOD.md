# Method and conditions

This document describes what the code does, stage by stage, so the paper's
description can be checked against it.

## Data and preprocessing

* Four NF-v3 datasets, 47 NetFlow features (identity and time columns are
  never read), 22 classes (see [DATA.md](DATA.md)).
* One fixed stratified 70/15/15 split per class within each dataset.
* Non-finite values are replaced by the pooled training median, then every
  feature is standardized with one scaler fitted on the pooled training rows
  of all four datasets. Validation and test rows never influence it.
* The dataset of origin of each row is kept as bookkeeping (losses, per-dataset
  reports). It is never an input of any model; every model's `forward` takes
  only the feature vector, which is asserted when a model is loaded.

## Shared training settings

Batch size 512; Adam with learning rate 1e-3 and no weight decay in every
stage; epochs: Stage A 30, Stage B 10, Stage C up to 30. Batches are drawn by
block-shuffling the training rows (no class-balanced sampler). Cross-entropy
uses square-root inverse-frequency class weights (normalized to mean 1),
computed on the rows the stage trains on.

## The dataset-expert MoE (M), 22,420 parameters

**Stage A, shared encoder.** An MLP encoder 47-128-64 (ReLU, dropout 0.2) is
trained on all pooled training rows with a temporary linear probe:
weighted cross-entropy plus 0.20 times a balanced supervised contrastive loss
on the encoder output (temperature 0.1). In the balanced loss every class
present in the batch contributes the mean similarity of its members to the
denominator, so frequent classes do not dominate the contrast. The probe is
discarded. Stage A runs for its full 30 epochs (no early stopping).

**Stage B, dataset experts.** The encoder is frozen. Four linear experts
64-22 (one per dataset) are trained one after another, each only on its own
dataset's training rows with that dataset's class weights, for 10 epochs.

**Stage C, gate.** Encoder and experts are frozen. A gate 64-32-4 (softmax
over experts) and a 4 x 22 class reliability matrix R (initialized to zero)
are trained on all pooled rows. For each row and class, the expert weights are
`softmax_over_experts(log gate + R[:, class])`; the class score is the weighted
sum of the experts' class probabilities, renormalized over classes. With R = 0
this is exactly the ordinary soft mixture. Loss: weighted negative log
likelihood + 0.1 x load-balance penalty (squared coefficient of variation of
mean gate use) + 1e-4 x mean(R squared). No dataset identity enters the loss.
The epoch with the best validation macro-F1 is kept (patience 5).

Note: during Stage C the frozen encoder is in training mode, so its dropout is
active while the gate trains (as in the development experiments); it is off
at evaluation and prediction.

**Prediction.** The gate sees only the encoder output; the class with the
highest mixed probability is predicted.

## Baselines

B2 and B3 start from M's Stage-A encoder of the same seed; B4 uses nothing from M.

* **B2 capacity-matched pooled MLP (22,419).** One classifier: M's frozen
  encoder plus a head 64-61-48-22. The two widths are chosen by a deterministic
  search for the total parameter count closest to M's. Stage B trains the
  head on four pooled exposure-matched streams; Stage C trains it on all rows
  with best-validation-epoch selection.
* **B3 hard-routing two-stage baseline (22,332).** Uses M's Stage-B experts.
  A router (64-32-4, no reliability matrix) is trained only to predict the
  dataset of origin (dataset-ID cross-entropy, no task-loss gradient, no load
  balancing). At prediction every row goes to exactly one expert, the router's
  argmax.
* **B4 pooled XGBoost (trees; no parameter count).** One gradient-boosted
  tree model (XGBoost, multiclass, trained on a GPU) on all pooled training
  rows, dataset-blind. Its size is reported as trees and leaves next to its
  results.

### B4 fairness contract

B4 gets nothing M does not get. Anything not listed is not used.

| Aspect | M | B4 |
|---|---|---|
| Training rows | all pooled training rows, natural class distribution | the same rows |
| Inputs | the 47 prepared features (median imputed, standardized), no dataset ID | the same prepared feature arrays, no dataset ID |
| Class weighting | square-root inverse frequency, normalized to mean 1, on the pooled training rows | the same weights from the same function, as per-row sample weights |
| Resampling | none | none |
| Loss | class-weighted cross-entropy (plus M's auxiliary terms) | class-weighted multiclass cross-entropy (softmax) |
| Model selection | best validation macro-F1 epoch (Stage C, patience 5 of at most 30 epochs) | best validation macro-F1 boosting round (checked every 10 rounds, patience 200 rounds, at most 2,000 rounds) |
| Regularization | dropout 0.2 | row and column subsampling 0.8, L2 penalty 1 on leaf values, minimum child weight 1 (fixed, not tuned) |
| Tuning on validation | settled in development (see below) | a grid of 4 configurations, once |
| Prediction | argmax of class probabilities | argmax of class probabilities |
| Hardware | one GPU | one GPU |

Not allowed for B4: dataset ID or anything derived from it, feature
engineering, oversampling or undersampling, `scale_pos_weight` or any class
weighting beyond the sample weights above, focal or custom losses, other
boosters (DART, linear), per-class thresholds, probability calibration. The
code sets the objective (`multi:softprob`) itself and rejects the
corresponding XGBoost settings.

Fixed XGBoost settings: `tree_method: hist` on the GPU, `max_bin` 256,
`grow_policy: lossguide` with `max_depth: 0` (leaf-wise growth limited by
`max_leaves`), `reg_lambda` 1, `min_child_weight` 1, `subsample` 0.8,
`colsample_bytree` 0.8; the seed is derived from the campaign seed (this is
what makes B4's seeds differ). Everything else is at XGBoost's defaults.

The L2 penalty on leaf values matters on these data: without it (a penalty of
0, LightGBM's default), leaves whose rows are already confidently classified
have near-zero hessians and output very large values, and multiclass training
diverges within a few tens of rounds. XGBoost's default of 1 is kept. The patience of 200 rounds (10% of the 2,000-round cap)
leaves room for the noise of validation macro-F1 between checks.

### How each condition targets macro-F1

No condition optimizes macro-F1 directly (it is not differentiable). Every
condition, M and B4 included, minimizes a class-weighted cross-entropy with
the same square-root inverse-frequency weights, and macro-F1 enters only
through those weights and through selection on validation macro-F1 (M's
Stage-C epoch, B4's boosting round and B4's tuning grid). No condition uses
per-class thresholds or post-hoc calibration in the main results.

Supplementary analysis, applied identically to every condition and seed
(`scripts/bias_tuning.py`): one additive offset per class on the log class
probabilities (Benign fixed at 0) is fitted on validation by coordinate ascent
over a fixed grid (-8 to 8 in steps of 0.25, up to 3 passes) to maximize
macro-F1, and applied unchanged to test. The fit uses every validation row of
classes with at most 200,000 validation rows and 200,000 rows of larger
classes, each weighted by n_c / k_c so the weighted confusion matrix estimates
the full validation one. It is reported separately from the main results
(`summary/Bias_Tuning_*.csv`); test data is never used for fitting.

Tuning protocol (pre-registered in `configs/baselines/B4_pooled_xgboost.yaml`):

* Grid: `max_leaves` in {63, 255} x `learning_rate` in {0.05, 0.1}, trained
  once with seed 0 on the fixed split. The configuration with the highest
  validation macro-F1 is used unchanged for every seed (ties go to the earlier
  configuration). Test data is never read during tuning.
* Before the grid, a timing pilot (20 rounds of the largest configuration on
  all training rows) records seconds per round, GPU memory and the
  worst-case projected time in `runs/B4/tuning/pilot.json`; it does not
  change any setting.
* If the chosen configuration's best round lies within the last 200 rounds of
  the 2,000-round cap, the seed runs use a 4,000-round cap instead, so training
  length never limits B4.
* Every B4 run trains in one pass and saves a checkpoint every 200 rounds.
  A run interrupted by a disconnect resumes from its last checkpoint; the
  resume round is recorded (`resumed_after_rounds` in the run's
  `progress.json` and in `Resource_Accounting.csv`), because row and column
  subsampling restarts there, so a resumed run is not bit-identical to an
  uninterrupted one.

## Ablations

* **A1 no shared encoder, capacity matched (22,284).** M without the shared
  encoder: the gate and the four dataset experts read the 47 standardized
  features directly (an identity encoder with no parameters), so there is no
  Stage A and no representation learning shared across datasets. The
  parameters M spends on its encoder (14,400) go to the experts instead: each
  expert is 47-73-22 (one hidden layer, ReLU, dropout 0.2; 5,132 parameters)
  instead of M's linear 64-22 (1,430). The gate keeps M's shape on the wider
  input, 47-32-4 (1,668), and R is unchanged (4 x 22 = 88). Total
  4 x 5,132 + 1,668 + 88 = 22,284, 0.6% below M's 22,420 (a hidden width of
  73 is the closest; 74 would give 22,580, 0.7% above). Forward cost: 21,780
  multiply-accumulates per row against M's 22,016.
  Stage B is M's: each expert trains only on its own dataset's training rows,
  with that dataset's class weights, for 10 epochs, here on the standardized
  features. Stage C is M's: experts frozen, gate and R trained on all pooled
  rows with the same loss, load-balance weight 0.1, reliability penalty 1e-4,
  patience and selection rule. None of M's hyperparameters were retuned for
  A1; this was fixed before A1 was run. A1 reuses nothing from M and depends on
  no other condition. As in M, Stage C runs the frozen modules in training
  mode, so the experts' dropout (0.2, after the hidden layer) is active while
  the gate trains and off at evaluation and prediction; M's experts are
  linear, so for M this concerns only the encoder's dropout.
* **A2 no gating, uniform expert average (20,120).** M's Stage-A encoder and
  M's four Stage-B experts of the same seed, combined with fixed equal weights:
  `combined = (1/4) * sum_d softmax(expert_logits_d)`. No gate network, no
  reliability matrix R, no Stage-C training and no validation-based
  selection; this is M's mixture with the gate weights fixed at 1/4 and R = 0.
  It is not capacity matched: M's gate (2,212) and R (88) are absent and
  nothing is added elsewhere.
* **A3 private encoders (80,020).** Each expert owns a full 47-128-64 encoder
  and the gate uses a separate encoder of the same shape. All five encoders
  start from M's Stage-A weights; in Stage B each private encoder trains
  together with its linear head on its own dataset; Stage C is as in M (all
  encoders and heads frozen). A3 is **not** capacity matched (about 3.6 times
  M); its parameter count is reported next to its results.

Expected readings, written before the final runs: TODO (author).
* A1: TODO
* A2: TODO
* A3: TODO
* B4: TODO

## Choices made on validation data in development

The representation weight 0.20 and the reliability penalty 1e-4 were chosen
in development experiments by validation macro-F1, never by test results.
TODO (author): one sentence on the grid that was compared.

## Evaluation

* Headline and selection metric: macro-F1 over classes with support.
  Weighted-F1 is reported for reference only.
* Every run is evaluated on validation and test; test results are reported
  over all rows and per dataset of origin.
* ROC-AUC and PR-AUC (one-vs-rest) are computed from 4096-bin score
  histograms to bound memory on tens of millions of rows; the method is named
  in every output.
* Across seeds: mean, SD and a t-based 95% interval; paired differences from
  M on common seeds; per seed, a paired bootstrap of the macro-F1 difference
  (1,000 resamples of test rows) and McNemar's test on per-row correctness.
* B4 is evaluated by exactly the same code: its probabilities go into the same
  per-row prediction files, tables, aggregation and significance tests.
