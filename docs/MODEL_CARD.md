# Model card: dataset-expert MoE for NetFlow intrusion detection

## Intended use

Research on multi-dataset network intrusion detection: classifying NetFlow v3
flow records into benign traffic or one of 21 attack classes, and reproducing
the paper's comparisons. Not a production intrusion detection system: it has
not been evaluated on live traffic, under adversarial manipulation, or on
networks unlike the four source datasets.

## Inputs and outputs

* **Input:** one flow record with the 47 NF-v3 features listed in
  [DATA.md](DATA.md) (raw values; the bundle applies the training-set median
  imputation and standardization). No dataset label or address is needed or
  accepted.
* **Output:** the predicted class (one of 22), its probability, optionally all
  22 probabilities, and for MoE models the 4 gate weights and the top expert.

## Bundles

Each trained run exports a bundle (`model.safetensors`, `config.json`,
`preprocessing.json`, `labels.json`, `metadata.json`); `load_bundle` verifies
the SHA-256 of every file and loads weights with safetensors only.
`metadata.json` records condition, seed, split, selected epochs, validation
and test macro-F1, parameters, code commit and library versions.

TODO: Hugging Face repository (`hub.repo_id`) once published; then
`load_bundle("hf:recommended")` and `hf:<condition>/seed<k>` work.

## Training data

Pooled training split (70% per class within each dataset) of NF-UNSW-NB15-v3,
NF-ToN-IoT-v3, NF-BoT-IoT-v3 and NF-CSE-CIC-IDS2018-v3. Training procedure:
[METHOD.md](METHOD.md).

## Evaluation

Test split (15% per class within each dataset), macro-F1 as the headline
metric, reported over all rows and per dataset.

Results (mean and SD over 5 seeds, and the recommended seed's numbers):
TODO: generated from `summary/` after the final runs.

## Selection

Stage C keeps the epoch with the best validation macro-F1. The recommended
bundle of a condition is its seed with the best validation macro-F1 (ties to
the lower seed). Test data never influences any choice.

## Limitations

* Trained and evaluated only on the four NF-v3 datasets, which come from lab
  testbeds; performance on other networks is unknown.
* Classes are defined by mapping each dataset's labels to a shared taxonomy
  (see DATA.md); merged classes (for example DoS) group attacks that may
  differ between datasets.
* Some classes are rare; their per-class scores have high variance across seeds.
* ROC-AUC and PR-AUC are histogram approximations (4096 bins).

## License and citation

Code: MIT ([LICENSE](../LICENSE)). Trained weights: TODO, decide their licence
after checking the terms of the four datasets they were trained on. The
datasets keep their own licences. Citation: [CITATION.cff](../CITATION.cff) (TODO: paper reference).
