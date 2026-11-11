# Reproducing the experiments

## The campaign

`configs/campaign.yaml` defines the whole experiment: seven conditions, five
seeds, the split policy and which stages each condition reuses.

| Condition | Trains | Reuses from M (same seed) |
|---|---|---|
| M | Stage A, B, C | |
| B1 generic MoE | Stage B, C | Stage A |
| B2 capacity-matched pooled MLP | Stage B, C (head only) | Stage A |
| B3 hard-routing two-stage baseline | Stage C (router only) | Stage A, Stage B |
| A1 experts trainable in Stage C | Stage C | Stage A, Stage B |
| A2 no class-conditional reliability | Stage C | Stage A, Stage B |
| A3 private encoders | Stage B, C | Stage A (copied into every encoder) |

One *unit* is one (condition, seed). Units run seed by seed and, within a seed,
in the order above, so M of a seed always exists before the conditions that
reuse it. Sharing M's Stage A gives every condition of a seed the same encoder
starting point, so differences come from what each condition changes.

```bash
python scripts/run_campaign.py                              # plan: what is done, what is pending
python scripts/run_campaign.py --execute                    # train everything pending
python scripts/run_campaign.py --only M --seeds 3 4 --execute
python scripts/run_campaign.py --only B1 B2 B3 --execute --runs-per-session 1
```

Without `--execute` nothing is trained. A unit whose parent is missing is
reported by name (for example `needs M seed 3 Stage B: run that first`).

## Seeds

* The default list is `seeds: [0, 1, 2, 3, 4]` in `configs/campaign.yaml`.
* Any list works on the command line or in the notebooks (`--seeds 3 4`,
  `SEEDS = [5, 6]`). Adding seeds later trains only the new ones.
* A unit counts as complete when `runs/<condition>/seed<k>/manifest.json`
  says so and its recorded config hash equals the current one. Changing any
  setting that affects results makes the affected units pending again; paths
  and logging settings do not.
* Aggregation uses whatever seeds are complete, prints n per condition, warns
  when conditions have different seed sets, and pairs conditions only on the
  seeds both have.

## The fixed split

All conditions and seeds use one train/validation/test split
(`split: {mode: fixed, split_seed: 0}`): within each dataset, each class is
divided 70/15/15 (classes with fewer than 3 rows go to train). The training
seed therefore changes only weight initialization, batch order and dropout,
and every comparison is made on identical test rows. The seed-to-seed
standard deviation measures training variability, not split variability.

`mode: per_seed` (split seed equal to training seed) is available in
`campaign.yaml` for a split-variability study.

The split is built once (two streaming passes over each CSV) and cached under
`OUTPUT_ROOT/cache/splits/`. Its signature covers the CSV contents (SHA-256),
the label mapping, the features, the fractions and the split seed, so
identical files produce identical splits on any machine. Each session copies
the cache to fast local disk (`SCRATCH_ROOT`). One scaler is fitted on the
pooled training rows only (enforced by an assertion) and reused by every run.

Runs trained on different splits are never mixed: aggregation and full
re-evaluation refuse them.

## Determinism

Every run seeds Python, NumPy and PyTorch, uses deterministic cuDNN and
`torch.use_deterministic_algorithms(True)`, and draws batch orders from NumPy
generators seeded per stage and epoch.

**Guarantee:** the same code, data, seeds, library versions and GPU model give
identical weights and metrics. A different GPU model, CUDA or cuDNN version can
change the last decimals because floating-point reductions may run in a
different order. The GPU, CUDA, cuDNN and library versions are recorded in
every run manifest and bundle. Evaluation, aggregation (bootstrap seed
`aggregation_seed: 0`) and prediction need no seed.

## Selection rules

* Stage C keeps the epoch with the best **validation** macro-F1 (patience 5).
* The recommended seed of each condition is the one with the best validation
  macro-F1 (ties go to the lower seed). Only M's is advertised.
* The test split is never used for any choice.

## Colab

Run the notebooks in order: `01_train_method`, then `02_train_baselines` and
`03_train_ablations`, then `04_evaluate` with `MODE = "tables"`.

* Choose a GPU runtime for 01 to 03.
* Edit only the SETTINGS cell. `RUNS_PER_SESSION = 1` trains one unit per
  execution; `None` trains until done or disconnected. Rerunning the notebook
  resumes where it stopped (completed units are skipped, an interrupted unit
  resumes from its last finished epoch).
* Output streams in the notebook and into `runs/<condition>/seed<k>/log.txt`.

**Colab secrets** (key icon in the left sidebar), all optional:

| Secret | Used for |
|---|---|
| `NIDS_DATA_DIR` | data folder, if not `MyDrive/NIDS_datasets` |
| `NIDS_OUTPUT_DIR` | output folder, if not `MyDrive/NIDS_outputs` |
| `GITHUB_TOKEN` | only to clone a private fork |
| `HF_TOKEN` | only to publish bundles to the Hugging Face Hub |

## Compute

TODO: wall time per unit for each condition on the GPU used for the paper,
and the total for 7 conditions x 5 seeds. `eval/Resource_Accounting.csv` of
each run records training and inference time.

## After training

```bash
python scripts/aggregate.py              # OUTPUT_ROOT/summary/
python scripts/make_paper_tables.py      # OUTPUT_ROOT/paper/
python scripts/sync_results.py           # copy small result files into results/ and paper/
python scripts/publish_hub.py            # dry run of the Hugging Face upload (needs hub.repo_id)
```
