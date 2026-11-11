"""Repository hygiene check: nothing personal or secret may be committed.

Scans tracked text files (or the files given) and fails on: GitHub and
Hugging Face tokens, email addresses (allowed only in ``CITATION.cff``),
home-directory paths, Google Drive desktop-client paths, Google Drive file
or folder links, Drive mount paths other than the two generic notebook
defaults, absolute paths inside committed JSON/CSV results, notebooks with
saved outputs or non-``None`` storage roots, and the em dash character.

All patterns are generic; this file contains no personal value.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass

EM_DASH = "—"
ALLOWED_DRIVE_PATHS = {
    "/content/drive", "/content/drive/MyDrive",
    "/content/drive/MyDrive/NIDS_datasets", "/content/drive/MyDrive/NIDS_outputs",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".csv", ".tex",
    ".ipynb", ".sh", ".cff", ".gitignore", "",
}
PATTERNS = [
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("macOS home path", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    ("Linux home path", re.compile(r"/home/(?!runner\b)[A-Za-z0-9._-]+/")),
    ("Google Drive desktop path", re.compile(r"GoogleDrive-|Library/CloudStorage")),
    ("Google Drive link", re.compile(r"drive\.google\.com/(?:file/d|drive/folders|open\?id=)")),
    ("Windows user path", re.compile(r"[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9._-]+")),
]
DRIVE_PATH = re.compile(r"/content/drive(?:/[^\s\"'`),\]]*)?")
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.])/(?:content|tmp|private|var|mnt|data|root|opt|srv)/[^\s\"',]+")
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".ipynb_checkpoints"}
# The checker necessarily contains the patterns it looks for.
SELF_FILES = {"src/dmoe/hygiene.py"}


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.excerpt}"


def tracked_files(root: str) -> list[str]:
    """Files git would commit (tracked plus untracked-but-not-ignored)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout
        return sorted(line for line in out.splitlines() if line)
    except (OSError, subprocess.CalledProcessError):
        files = []
        for directory, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            for name in names:
                files.append(os.path.relpath(os.path.join(directory, name), root))
        return sorted(files)


def _excerpt(text: str) -> str:
    text = text.strip()
    return text if len(text) <= 120 else text[:117] + "..."


def check_text(path: str, text: str) -> list[Finding]:
    findings = []
    is_citation = os.path.basename(path) == "CITATION.cff"
    is_result = path.replace(os.sep, "/").startswith(("results/", "paper/")) and path.endswith((".json", ".csv"))
    for number, line in enumerate(text.splitlines(), start=1):
        if EM_DASH in line:
            findings.append(Finding(path, number, "em dash (U+2014)", _excerpt(line)))
        for kind, pattern in PATTERNS:
            if kind == "email address" and is_citation:
                continue
            if pattern.search(line):
                findings.append(Finding(path, number, kind, _excerpt(line)))
        for match in DRIVE_PATH.finditer(line):
            value = match.group(0).rstrip("/.")
            if value not in ALLOWED_DRIVE_PATHS:
                findings.append(Finding(path, number, "Drive path outside the generic defaults", value))
        if is_result and ABSOLUTE_PATH.search(line):
            findings.append(Finding(path, number, "absolute path in a committed result file", _excerpt(line)))
    return findings


def check_notebook(path: str, text: str) -> list[Finding]:
    findings = []
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as exc:
        return [Finding(path, 0, "notebook is not valid JSON", str(exc))]
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            findings.append(Finding(path, index, "notebook cell has saved outputs", f"cell {index}"))
        if cell.get("execution_count") is not None:
            findings.append(Finding(path, index, "notebook cell has an execution count", f"cell {index}"))
        source = "".join(cell.get("source", []))
        for name in ("DATA_ROOT", "OUTPUT_ROOT"):
            match = re.search(rf"^{name}\s*=\s*(.+?)(?:\s*#.*)?$", source, flags=re.M)
            if match and match.group(1).strip() != "None":
                findings.append(Finding(path, index, f"{name} must default to None", match.group(0)))
    return findings


def check_files(root: str, files: list[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for relative in files if files is not None else tracked_files(root):
        relative = relative.replace(os.sep, "/")
        if relative in SELF_FILES:
            continue
        suffix = os.path.splitext(relative)[1].lower()
        if suffix not in TEXT_SUFFIXES:
            continue
        path = os.path.join(root, relative)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except UnicodeDecodeError:
            continue
        findings.extend(check_text(relative, text))
        if suffix == ".ipynb":
            findings.extend(check_notebook(relative, text))
    return findings
