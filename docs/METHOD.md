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
* **B4 pooled LightGBM (trees; no parameter count).** One gradient-boosted
  tree model (LightGBM, multiclass, CPU) trained on a fixed class-capped,
  reweighted sample of the pooled training rows (see below), dataset-blind. Its size is reported as trees and leaves next to its results.

### B4 fairness contract

B4 gets nothing M does not get. Anything not listed is not used.

| Aspect | M | B4 |
|---|---|---|
| Training rows | all pooled training rows, natural class distribution | one fixed class-capped sample of the same rows (about 10%), reweighted so each class keeps its total weight (below) |
| Inputs | the 47 prepared features (median imputed, standardized), no dataset ID | the same prepared feature arrays, no dataset ID |
| Class weighting | square-root inverse frequency, normalized to mean 1, on the pooled training rows | the same weights from the same function on all pooled training rows, as per-row sample weights, times the sampling correction n_c / k_c |
| Resampling | none | none beyond the declared training sample |
| Model selection | best validation macro-F1 epoch (Stage C, patience 5) | best validation macro-F1 boosting round (checked every 10 rounds, patience 50 rounds, at most 2,000 rounds) |
| Stochastic regularization | dropout 0.2 | row and feature subsampling 0.8 (fixed, not tuned) |
| Tuning on validation | settled in development (see below) | a grid of 4 configurations, once |
| Prediction | argmax of class probabilities | argmax of class probabilities |

Not allowed for B4: dataset ID or anything derived from it, feature
engineering, oversampling or undersampling other than the declared training
sample, `is_unbalance` or `class_weight`
beyond the sample weights above, focal or custom losses, DART, per-class
thresholds, probability calibration. The code rejects the corresponding
LightGBM settings.

Fixed LightGBM settings: `objective: multiclass`, `max_bin` 255,
`feature_fraction` 0.8, `bagging_fraction` 0.8, `bagging_freq` 1,
`deterministic: true`, `force_row_wise: true`, all CPU cores; the subsampling
seeds are derived from the campaign seed (this is what makes B4's seeds
differ). Everything else is at LightGBM's defaults.

Training sample. On all 46.9M pooled training rows, a 22-class LightGBM run
did not fit in the memory of a Colab CPU high-RAM runtime (it keeps a score,
a gradient and a hessian per row and class), and its training time would far
exceed the compute budget of the other conditions. Every B4 run (tuning and all seeds) uses one fixed sample of about 10% of the
training rows, drawn once with seed 0:

* A per-class cap K is the largest count for which the sample stays within
  10% of the training rows. Classes with at most K rows keep every row;
  larger classes keep K rows drawn uniformly without replacement.
* A kept row of class c with n_c training rows, k_c of them kept, gets M's
  class weight times n_c / k_c. Each class therefore keeps exactly its total
  weight under M's loss, and the weighted training loss is an unbiased
  estimate of the loss M trains on. Classes kept whole get exactly M's weight.
* Validation and test are complete; selection and evaluation do not change.
* The per-class counts, kept rows and multipliers are recorded in
  `runs/B4/tuning/training_sample.json`.

This keeps every row of the rare attack classes that dominate macro-F1 and
thins only the large classes.

Tuning protocol (pre-registered in `configs/baselines/B4_pooled_lightgbm.yaml`):

* Grid: `num_leaves` in {63, 255} x `learning_rate` in {0.05, 0.1}, trained
  once with seed 0 on the fixed split. The configuration with the highest
  validation macro-F1 is used unchanged for every seed (ties go to the earlier
  configuration). Test data is never read during tuning.
* Time limit: before the grid, a timing pilot (20 rounds of the largest
  configuration on the training sample) projects the whole grid. Above 24
  hours, tuning stops before training anything and the setting is revisited
  by the author; the pilot's measurements are recorded in
  `runs/B4/tuning/pilot.json`. The grid and the seed runs use the same
  training sample.
* If the chosen configuration's best round lies within the last 50 rounds of
  the 2,000-round cap, the seed runs use a 4,000-round cap instead, so training
  length never limits B4.
* Every B4 run trains in chunks of 200 rounds, each continued from the saved
  model, so a run that is interrupted and resumed follows exactly the same
  path as an uninterrupted one.

## Ablations

* **A1 no contrastive loss (22,420).** M with a cross-entropy-only Stage A:
  the balanced supervised contrastive term is dropped; encoder, probe, class
  weights, 30 epochs, optimizer and batch order are as in M (the contrastive
  loss has no parameters, so A1 and M start from identical weights). A1 trains
  its own Stage A, B and C per seed (about one M run). Stage B and Stage C use
  exactly M's hyperparameters (load-balance weight 0.1, reliability penalty
  1e-4, epochs, patience, selection rule); they were not retuned for A1. This
  choice was fixed before A1 was run.
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
