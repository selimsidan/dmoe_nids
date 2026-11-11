# dmoe_nids: dataset-expert mixture of experts for network intrusion detection

One intrusion detection classifier trained on four public NetFlow datasets at
once (NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3, NF-CSE-CIC-IDS2018-v3).
A shared encoder feeds one expert per dataset; a gate, which never sees the
dataset of origin, mixes the experts per class. The model is a single
classifier that needs no dataset label at prediction time. The paper asks
whether this recovers the per-dataset accuracy that naive pooling loses.
TODO: one-sentence claim from the final results.

This repository contains the code, configurations and notebooks that produce
every number, table and figure of the paper (TODO: paper reference).

| | |
|---|---|
| Method | dataset-expert MoE (condition **M**) |
| Baselines | **B1** generic MoE, **B2** capacity-matched pooled MLP, **B3** hard-routing two-stage baseline |
| Ablations | **A1** experts trainable in Stage C, **A2** no class-conditional reliability, **A3** private encoders |
| Seeds | 5 per condition (0 to 4), one fixed train/validation/test split |
| Headline metric | test macro-F1 (selection always on validation macro-F1) |

## Results

Results tables are generated from the result files once the campaign has run
(`scripts/aggregate.py`, `scripts/make_paper_tables.py`); none are written by hand.
TODO: insert the generated main table after the final runs.

## Install

```bash
git clone https://github.com/selimsidan/dmoe_nids.git
cd dmoe_nids
pip install -r requirements.txt   # exact versions used for the paper
pip install -e . --no-deps
pytest                            # about 2 minutes on a CPU, synthetic data only
```

Python 3.10 or newer. A GPU is needed only for training.

## Quick start: predict with a trained model

```python
import pandas as pd
from dmoe.inference.bundle import load_bundle

bundle = load_bundle("outputs/runs/M/seed0/bundle")   # or "hf:recommended" (TODO: hub not published yet)
flows = pd.read_csv("my_netflow_v3_flows.csv")        # needs the 47 NF-v3 feature columns
predictions = bundle.predict(flows, with_probs=True, with_gate=True)
```

or from the command line:

```bash
python scripts/predict.py --bundle outputs/runs/M/seed0/bundle --input flows.csv --output predictions.csv --gate
```

No dataset identity is used for prediction; `--labels-from <dataset>` only maps
the CSV's `Attack` column to the 22 classes so the predictions can be scored.

## Reproduce the paper

1. **Data.** Download the four CSVs and arrange them as described in
   [docs/DATA.md](docs/DATA.md); check them with `python scripts/verify_data.py`.
2. **Train.** Either run the Colab notebooks in order (below) or, on any machine
   with a GPU:
   ```bash
   python scripts/run_campaign.py              # prints the plan
   python scripts/run_campaign.py --execute    # trains all 7 conditions x 5 seeds
   ```
   M is trained first for each seed; the baselines and ablations reuse its
   Stage A (and, for B3, A1, A2, its Stage B). Interrupted runs resume.
3. **Aggregate and build the tables.**
   ```bash
   python scripts/aggregate.py
   python scripts/make_paper_tables.py         # LaTeX, CSV and PDF in outputs/paper/
   ```

Details (seeds, the fixed split, determinism, dependencies between conditions,
compute) are in [docs/REPRODUCING.md](docs/REPRODUCING.md). What each stage and
condition does is in [docs/METHOD.md](docs/METHOD.md).

### Verify the paper's numbers without retraining

* **Quick (CPU, no data):** `python scripts/evaluate.py --predictions <run>/eval/test/Test_Predictions.npz --out check`
  recomputes every metric, table and figure from the saved per-row predictions
  (TODO: link to the release asset holding the prediction files).
* **Full:** `python scripts/evaluate.py --bundle <bundle> --split test --out check`
  rebuilds the test split from the CSVs and runs the model again.

## Colab notebooks

| Notebook | Purpose | GPU |
|---|---|---|
| [01_train_method](notebooks/01_train_method.ipynb) | train M (run first) | yes |
| [02_train_baselines](notebooks/02_train_baselines.ipynb) | train B1, B2, B3 | yes |
| [03_train_ablations](notebooks/03_train_ablations.ipynb) | train A1, A2, A3 | yes |
| [04_evaluate](notebooks/04_evaluate.ipynb) | re-evaluate runs; aggregate seeds; paper tables | no |
| [05_predict](notebooks/05_predict.ipynb) | predict any NF-v3 CSV with a bundle | no |

Open a notebook in Colab (TODO: Colab badges once the repository is final),
edit only its SETTINGS cell and choose *Run all*. By default data is read from
`MyDrive/NIDS_datasets` and everything is written to `MyDrive/NIDS_outputs`;
other folders can be typed in the SETTINGS cell or stored as the Colab secrets
`NIDS_DATA_DIR` and `NIDS_OUTPUT_DIR`.

## Repository map

```
configs/      base.yaml (shared settings), campaign.yaml (conditions, seeds, split, dependencies),
              method/ baselines/ ablations/ (one file per condition), sources.yaml, checksums.json
src/dmoe/     data/ (registry, split building, verification), models/, training/ (stages, campaign),
              evaluation/ (metrics, figures, aggregation, paper tables), inference/ (bundles), colab.py
scripts/      command-line entry points (thin wrappers around src/dmoe)
notebooks/    Colab notebooks 01 to 05
tests/        CPU tests on synthetic data
docs/         DATA, REPRODUCING, METHOD, RESULTS_SCHEMA, MODEL_CARD
results/      small result files copied from the runs (scripts/sync_results.py)
paper/        generated tables and figures
```

Everything generated (split cache, checkpoints, bundles, predictions, logs)
goes to `OUTPUT_ROOT` (default `./outputs`), never into the repository.

## Data

The datasets are not redistributed. They are the NF-v3 versions published by
the University of Queensland; see [docs/DATA.md](docs/DATA.md) for sources,
citations and licences.

## Citation

See [CITATION.cff](CITATION.cff). TODO: paper reference.

## License

MIT, see [LICENSE](LICENSE). The datasets keep their own licences.
