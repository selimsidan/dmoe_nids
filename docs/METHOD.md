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

All baselines start from M's Stage-A encoder of the same seed.

* **B1 generic MoE (22,420).** Same architecture, gate and reliability matrix
  as M, but Stage B trains each expert on a random stream of pooled rows whose
  length equals the i-th largest dataset (the same exposure M's experts get),
  with pooled class weights. No expert is ever assigned a dataset.
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

## Ablations

* **A1 experts trainable in Stage C (22,420).** M with `freeze_experts: false`
  in Stage C; the encoder stays frozen. Reuses M's Stage A and B.
* **A2 no class-conditional reliability (22,332).** A plain soft mixture: the
  same gate weights for every class, no R and no R penalty. Reuses M's Stage
  A and B.
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
