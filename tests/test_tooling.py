"""Data verification, hygiene, notebooks, Colab setup, result syncing and the CLI scripts."""
import json
import os
import subprocess
import sys

import nbformat
import pandas as pd
import pytest

from dmoe import colab, hygiene, paths
from dmoe.data.registry import FEATURE_COLUMNS
from dmoe.data.source_hash import file_digests
from dmoe.data.verify import count_rows, verify_data
from dmoe.release import sync_results

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_DIR = os.path.join(REPO, "notebooks")


# ---------------------------------------------------------------- data verification
def _tiny_dataset(root, name="NF-ToN-IoT-v3", rows=5, drop=None):
    path = paths.dataset_csv_path(name, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [c for c in [*FEATURE_COLUMNS, "Label", "Attack"] if c != drop]
    pd.DataFrame([[0] * (len(columns) - 1) + ["Benign"]] * rows, columns=columns).to_csv(path, index=False)
    return path


def _checksums(tmp_path, entry):
    path = tmp_path / "checksums.json"
    path.write_text(json.dumps({"NF-ToN-IoT-v3": entry}))
    return str(path)


def test_verify_passes_and_counts(tmp_path):
    csv = _tiny_dataset(str(tmp_path / "data"))
    assert count_rows(csv) == 5
    digests = file_digests(csv, ("sha256", "sha1"))
    sums = _checksums(tmp_path, {"rows": 5, **digests})
    [result] = verify_data(str(tmp_path / "data"), ["NF-ToN-IoT-v3"], checksums_path=sums)
    assert result.ok


@pytest.mark.parametrize("entry, drop, message", [
    ({"rows": 6}, None, "has 5 rows, expected 6"),
    ({"rows": 5}, "IN_BYTES", "lacks 1 expected column"),
    ({"rows": 5, "sha1": "0" * 40}, None, "SHA1"),
])
def test_verify_failures_point_to_docs(tmp_path, entry, drop, message):
    _tiny_dataset(str(tmp_path / "data"), drop=drop)
    [result] = verify_data(str(tmp_path / "data"), ["NF-ToN-IoT-v3"], checksums_path=_checksums(tmp_path, entry))
    assert not result.ok and any(message in p and "docs/DATA.md" in p for p in result.problems)


def test_verify_missing_file(tmp_path):
    [result] = verify_data(str(tmp_path / "nothing"), ["NF-ToN-IoT-v3"], checksums_path=_checksums(tmp_path, {}))
    assert not result.ok and "file not found" in result.problems[0]


# ---------------------------------------------------------------- hygiene
FORBIDDEN = {
    "token": "ghp_" + "A" * 36,
    "hf": "hf_" + "b" * 34,
    "email": "someone" + "@" + "example.org",
    "home": "/" + "Users/someone/data.csv",
    "drive client": "Google" + "Drive-x/My Drive",
    "drive path": "/content/drive/MyDrive/" + "private_folder",
    "dash": "a " + chr(0x2014) + " b",
}


def test_hygiene_clean_file_passes(tmp_path):
    (tmp_path / "ok.py").write_text("x = '/content/drive/MyDrive/NIDS_datasets'\n")
    assert hygiene.check_files(str(tmp_path), ["ok.py"]) == []


@pytest.mark.parametrize("kind", sorted(FORBIDDEN))
def test_hygiene_catches(tmp_path, kind):
    (tmp_path / "bad.md").write_text(f"text {FORBIDDEN[kind]}\n")
    assert hygiene.check_files(str(tmp_path), ["bad.md"])


def test_hygiene_citation_may_hold_email(tmp_path):
    (tmp_path / "CITATION.cff").write_text(f"email: {FORBIDDEN['email']}\n")
    assert hygiene.check_files(str(tmp_path), ["CITATION.cff"]) == []


def test_hygiene_result_files_must_be_relative(tmp_path):
    os.makedirs(tmp_path / "results")
    (tmp_path / "results" / "m.json").write_text('{"path": "/content/outputs/runs/M"}\n')
    assert hygiene.check_files(str(tmp_path), ["results/m.json"])


def test_hygiene_notebook_rules(tmp_path):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("DATA_ROOT = '/x'\nOUTPUT_ROOT = None")])
    notebook.cells[0].outputs = [nbformat.v4.new_output("stream", text="hi")]
    nbformat.write(notebook, tmp_path / "n.ipynb")
    kinds = {f.kind for f in hygiene.check_files(str(tmp_path), ["n.ipynb"])}
    assert "notebook cell has saved outputs" in kinds and "DATA_ROOT must default to None" in kinds


def test_repository_is_clean():
    findings = hygiene.check_files(REPO)
    assert findings == [], "\n".join(map(str, findings))


# ---------------------------------------------------------------- notebooks
NOTEBOOKS = ["01_train_method", "02_train_baselines", "03_train_ablations", "04_evaluate", "05_predict"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_structure(name):
    notebook = nbformat.read(os.path.join(NOTEBOOK_DIR, f"{name}.ipynb"), as_version=4)
    nbformat.validate(notebook)
    assert notebook.cells[0].cell_type == "markdown" and "docs/DATA.md" in notebook.cells[0].source
    settings = notebook.cells[1].source
    assert "SETTINGS: the only cell you edit" in settings
    namespace = {}
    exec(settings, namespace)  # the settings cell is plain assignments
    assert namespace["DATA_ROOT"] is None and namespace["OUTPUT_ROOT"] is None
    assert namespace["REPO_REF"] == "main" and namespace["STORAGE"] == "drive"
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, name, "exec")
    if name.startswith("0") and "train" in name:
        assert namespace["EXECUTE"] is False and namespace["RUNS_PER_SESSION"] == 1


# ---------------------------------------------------------------- colab roots
class FakeUserdata:
    def __init__(self, values, denied=()):
        self.values, self.denied = values, set(denied)

    def get(self, name):
        if name in self.denied:
            raise PermissionError("notebook access denied")
        if name not in self.values:
            raise KeyError(name)
        return self.values[name]


def test_colab_typed_over_secret_over_default():
    secrets = FakeUserdata({"NIDS_DATA_DIR": "/content/drive/MyDrive/" + "elsewhere"})
    assert colab.resolve_colab_root("data", "/typed", "drive", secrets) == colab.ColabRoot("/typed", "typed")
    assert colab.resolve_colab_root("data", None, "drive", secrets).source == "secret"
    assert colab.resolve_colab_root("output", None, "drive", secrets) == colab.ColabRoot(colab.DEFAULTS["drive"]["output"], "default")
    denied = FakeUserdata({}, denied={"NIDS_DATA_DIR"})
    assert colab.resolve_colab_root("data", None, "local", denied) == colab.ColabRoot("/content/data", "default")


def test_colab_setup_exports_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(colab, "SCRATCH", str(tmp_path / "scratch"))
    roots = colab.setup("local", str(tmp_path / "data"), str(tmp_path / "out"), require_data=False,
                        userdata=FakeUserdata({}))
    assert os.environ["NIDS_OUTPUT_DIR"] == str(tmp_path / "out") and roots["data"].source == "typed"
    with pytest.raises(FileNotFoundError, match="docs/DATA.md"):
        colab.setup("local", str(tmp_path / "missing"), str(tmp_path / "out"), userdata=FakeUserdata({}))


# ---------------------------------------------------------------- results sync and CLI
def test_sync_results_copies_only_small_relative_files(trained, tmp_path):
    from dmoe.evaluation.aggregate import aggregate

    aggregate(trained["output"], resamples=20, figures=False)
    written = sync_results(trained["output"], str(tmp_path), figures=False)
    assert written
    for path in written:
        assert os.path.splitext(path)[1] in {".csv", ".json", ".tex", ".pdf", ".png"}
        if path.endswith((".csv", ".json")):
            assert trained["output"] not in open(path).read()
    assert os.path.isfile(tmp_path / "results" / "splits" / "split.json")
    assert not any("checkpoints" in p or "bundle" in p for p in written)
    assert hygiene.check_files(str(tmp_path), [os.path.relpath(p, tmp_path) for p in written]) == []


def _script(name, *args, env=None):
    return subprocess.run([sys.executable, os.path.join(REPO, "scripts", name), *args],
                          capture_output=True, text=True, env={**os.environ, **(env or {})}, cwd=REPO)


def test_cli_plan_and_verify(trained):
    from synthetic import FAST_OVERRIDES

    roots = ["--data-root", trained["data"], "--output-root", trained["output"], "--scratch-root", trained["scratch"]]
    sets = [arg for override in FAST_OVERRIDES for arg in ("--set", override)]
    plan = _script("run_campaign.py", "--only", "M", "B1", "--seeds", "0", "1", *roots, *sets)
    assert plan.returncode == 0, plan.stderr
    assert "M seed 0: complete" in plan.stdout and "B1 seed 1: pending" in plan.stdout and "dry run" in plan.stdout
    verify = _script("verify_data.py", "--no-hash", *roots)
    assert verify.returncode == 1 and "docs/DATA.md" in verify.stdout  # synthetic row counts differ


def test_cli_predict(trained, tmp_path):
    source = paths.dataset_csv_path("NF-UNSW-NB15-v3", trained["data"])
    out = tmp_path / "pred.csv"
    result = _script("predict.py", "--bundle", os.path.join(trained["output"], "runs", "M", "seed0", "bundle"),
                     "--input", source, "--output", str(out), "--gate", "--labels-from", "NF-UNSW-NB15-v3")
    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(out)) == len(pd.read_csv(source))
    assert os.path.isfile(tmp_path / "pred_evaluation" / "Overall_Metrics.csv")


def test_no_source_file_is_gitignored():
    """Every file that belongs in the repository must be committable (none hidden by .gitignore)."""
    if not os.path.isdir(os.path.join(REPO, ".git")):
        pytest.skip("not a git checkout")
    wanted = []
    for top in ("src", "configs", "scripts", "docs", "notebooks", "tests"):
        for directory, dirs, files in os.walk(os.path.join(REPO, top)):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.endswith(".egg-info")]
            wanted += [os.path.relpath(os.path.join(directory, f), REPO) for f in files if not f.endswith(".pyc")]
    result = subprocess.run(["git", "check-ignore", "--no-index", *wanted], cwd=REPO, capture_output=True, text=True)
    assert result.stdout.strip() == "", f"ignored by .gitignore:\n{result.stdout}"
