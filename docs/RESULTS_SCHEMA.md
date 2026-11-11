# Result files

All paths are relative to `OUTPUT_ROOT`. Small files are copied into the
repository by `scripts/sync_results.py` (`results/runs/...`,
`results/summary/...`, `paper/...`); data, caches, checkpoints, bundles,
per-row predictions and logs are never committed.

Conventions:

* Macro averages are over classes with support in the rows evaluated.
* Micro precision, recall and F1 all equal accuracy for single-label
  multiclass prediction; they are listed for completeness.
* Weighted averages weight each class by its support; reference only.
* `auc_method` names how ROC-AUC and PR-AUC were computed
  (`histogram_approximation_<bins>_bins`).
* `origin` is the dataset a row came from; `ALL` means all rows.

## One run: `runs/<condition>/seed<k>/`

```
manifest.json          status, config hash, split, environment, checkpoint and bundle hashes, results
log.txt                streamed training output
Training_History.csv   per-epoch training record of all stages (shared stages included)
figures/               Stage_{A,B,C}_Learning_Curves.png
checkpoints/           run_contract.json, stage_A/, stage_B/, stage_C/ (not committed)
bundle/                the exported model (not committed; see MODEL_CARD.md)
eval/Resource_Accounting.csv
eval/selection/        validation reports written during Stage C selection
eval/val/, eval/test/  tables and figures below, plus Val_/Test_Predictions.npz
```

### `eval/<split>/Overall_Metrics.csv`
One row for `ALL` and one per origin dataset.
`condition, seed, split, origin, macro_f1, macro_precision, macro_recall,
micro_f1, micro_precision, micro_recall, weighted_f1, weighted_precision,
weighted_recall, accuracy, balanced_accuracy, matthews_corrcoef, cohen_kappa,
roc_auc_ovr_macro, roc_auc_ovr_weighted, roc_auc_ovr_micro, pr_auc_ovr_macro,
pr_auc_ovr_weighted, pr_auc_ovr_micro, foreign_prediction_rate,
classes_with_support, rows, auc_method`

* `balanced_accuracy`: mean recall over classes with support.
* `foreign_prediction_rate`: share of rows predicted as a class that does not
  occur in that origin dataset (for example a ToN-IoT row predicted as
  Exploits). On the `ALL` row it counts classes that occur in no dataset, so
  it is 0 whenever all 22 classes are present.

### `eval/<split>/Per_Dataset_Metrics.csv`
The per-origin rows of `Overall_Metrics.csv` plus `dataset` (display name).

### `eval/<split>/Per_Class_Metrics.csv`
One row per class, for `ALL` and for each origin.
`condition, seed, split, origin, class, origin_datasets, support,
predicted_support, precision, recall, f1, specificity, false_positive_rate,
false_negative_rate, tp, fp, fn, tn, prevalence, roc_auc_ovr, pr_auc_ovr,
auc_method` (`origin_datasets`: the datasets where the class occurs, `|`-separated).

### Confusion matrices
`Confusion_Matrix_ALL_*` (all 22 classes) and `Confusion_Matrix_<dataset>_*`
(classes present in that dataset), each as:

* `_long.csv`: `true_class, predicted_class, count, row_ratio, column_ratio`
* `_counts.csv`: rows true class, columns predicted class, counts
* `_row_normalized.csv`: the same divided by row totals
* `figures/<name>.png|pdf`: true classes as rows; each cell shows the count
  over the row percentage, empty cells are blank, colour is the row percentage.

### Routing (MoE conditions)
* `Gate_By_Dataset.csv`: `origin, rows, own_expert_top_fraction,
  mean_gate_entropy, mean_max_gate, mean_gate__<expert>...`
  (`own_expert_top_fraction` only for dataset-named experts).
* `Expert_Utilization.csv`: `expert, argmax_rows, argmax_fraction,
  argmax_fraction__<origin>...`
* `Route_Confusion.csv`: `origin, top_expert, count, row_ratio`
  (and `Route_Confusion_counts.csv` as a matrix).
* `Expert_Performance_By_Dataset.csv`: every expert evaluated alone on every
  origin: `origin, expert, is_assigned_expert` and the overall metrics
  without AUCs.
* `Class_Conditional_Reliability.csv` (M, B1, A1, A3): `expert, class,
  reliability_logit, reliability_only_expert_preference` (softmax of R over
  experts for that class).
* `figures/Gate_Mass_Heatmap`, `figures/Route_Confusion`,
  `figures/Class_Reliability_Heatmap`.

### `eval/<split>/<Split>_Predictions.npz` (not committed)
`y_true` int16, `y_pred` int16, `dataset_idx` uint8, `confidence` float16,
`gate` float16 (rows x experts, MoE only), `roc_positive`/`roc_negative`
int64 (datasets x classes x bins score histograms), `expert_confusion` int64
(datasets x experts x classes x classes), `reliability` float32 (when
present), `header` (JSON: class, dataset and expert names, per-dataset class
lists, split and inference metadata). Enough to recompute every table and
figure above with `scripts/evaluate.py --predictions`.

### `eval/Resource_Accounting.csv`
Parameters (`total_parameters`, `trainable_parameters` in Stage C,
`encoder_parameters`, `gate_router_parameters` (gate plus R),
`classifier_expert_parameters`), compute (`stored_linear_macs`,
`active_parameters_per_sample_*`, `forward_macs_per_sample_*`,
`forward_flops_per_sample_*`; FLOPs = 2 x MACs of linear layers,
`flop_convention`), B2's matching (`matching_*`, `resolved_dense_hidden_dims`),
training cost per stage (`<A|B|C>_optimizer_steps, _examples_seen,
_epochs_completed, _wall_seconds, _selected_epoch`, totals), and test
inference cost (`test_inference_wall_seconds`, `test_inference_rows_per_second`,
`test_inference_microseconds_per_row`, `test_inference_device`,
`test_peak_memory_bytes`).

### `Training_History.csv`
`seed, stage, encoder_role, dataset, epoch, objective, learning_rate, rows,
optimizer_steps, examples_seen, epoch_seconds, train_total_loss,
train_ce_loss, train_representation_loss, train_weighted_representation_loss,
train_representation_fraction, train_balance_penalty, train_dataset_aux_loss,
train_anchor_penalty, train_reliability_penalty, val_macro_f1,
best_val_macro_f1, improved, patience_left`

### `eval/selection/`
`Validation_Overall_Metrics.csv`, `Validation_Per_Dataset_Metrics.csv`,
`Validation_Per_Class_Metrics.csv` (macro and weighted F1, macro precision
and recall, accuracy, support) and, for dataset-expert conditions,
`Validation_Expert_Owned_StageC.csv` (each expert on its own dataset).

## Across seeds: `summary/`

* `Summary_Overall.csv`, `Summary_Per_Dataset.csv` (`origin`),
  `Summary_Per_Class.csv` (`class`): `condition, [total_parameters | origin |
  class], metric, mean, sd, min, max, n, seeds, ci95_low, ci95_high`
  (SD with ddof 1; t-based 95% interval of the mean).
* `Paired_vs_M.csv`: `reference, condition, origin, metric, difference_mean,
  difference_sd, difference_min, difference_max, difference_ci95_low,
  difference_ci95_high, n_common_seeds, common_seeds, seeds_favoring_reference`
  (difference = M minus condition, on common seeds only).
* `Paired_vs_M_Per_Seed.csv`: `reference, condition, origin, metric, seed, difference`.
* `Significance.csv`: per common seed, `observed_difference` of test macro-F1
  (M minus condition), `bootstrap_ci95_low/high`, `bootstrap_p_two_sided`
  (paired row bootstrap, `bootstrap_resamples`), and McNemar's test
  (`mcnemar_only_reference_correct`, `mcnemar_only_condition_correct`,
  `mcnemar_statistic`, `mcnemar_p`, `mcnemar_method`: exact binomial below 25
  discordant rows, otherwise chi-square with continuity correction).
* `Recommended_Seeds.csv`: `condition, recommended_seed, val_macro_f1, rule,
  seeds_considered, advertised`.
* `Summary_Info.json`: seeds per condition, warnings, split signatures.
* `confusion/<condition>/`: test confusion matrix summed over seeds.
* `training/<condition>/`: learning curves across seeds (mean and SD band).

## Paper exhibits: `paper/`

`Table_Main`, `Table_Ablations`, `Table_Per_Dataset`, `Table_Per_Class`
(`.tex` and `.csv` with unrounded numbers), `Figure_Routing_<condition>` and
`Figure_Confusion_<condition>` (`.pdf`, `.png`).

## Data cache: `cache/` (not committed)

`splits/<dataset>/split-<signature>/` (memory-mapped `.npy` arrays and
`metadata.json`), `preprocessors/pooled-<signature>.npz` (scaler mean, scale,
train medians), `split.json` (split description, committed as
`results/splits/split.json`), `source_digests.json` (cached file hashes).
