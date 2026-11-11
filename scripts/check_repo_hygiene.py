"""Fail if any committable file contains tokens, personal paths, emails or em dashes.

Usage: python scripts/check_repo_hygiene.py [files ...]
Without arguments it checks every file git would commit.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dmoe.hygiene import check_files  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(argv: list[str]) -> int:
    files = [os.path.relpath(os.path.abspath(f), ROOT) for f in argv] or None
    findings = check_files(ROOT, files)
    for finding in findings:
        print(finding)
    if findings:
        print(f"hygiene check FAILED: {len(findings)} finding(s)")
        return 1
    print("hygiene check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
