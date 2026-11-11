"""Resolution of ``hf:`` bundle references against the Hugging Face Hub.

``hf:<condition>/seed<k>`` names one bundle and ``hf:recommended`` the
recommended seed of the method (M). The hub repository id comes from
``configs/base.yaml`` (``hub.repo_id``) or the ``NIDS_HUB_REPO`` environment
variable; it is empty until the bundles are published.
"""
from __future__ import annotations

import os

from .training.config import BASE_CONFIG, load_yaml

RECOMMENDED_POINTER = "recommended.json"


def hub_repo_id() -> str:
    repo = os.environ.get("NIDS_HUB_REPO") or (load_yaml(BASE_CONFIG).get("hub", {}) or {}).get("repo_id")
    if not repo:
        raise RuntimeError(
            "No Hugging Face repository is configured yet (hub.repo_id in configs/base.yaml or NIDS_HUB_REPO). "
            "Use a local bundle path instead."
        )
    return str(repo)


def parse_reference(reference: str) -> tuple[str, int | None]:
    if not reference.startswith("hf:"):
        raise ValueError(f"not a hub reference: {reference!r}")
    body = reference[3:]
    if body == "recommended":
        return "recommended", None
    condition, _, seed = body.partition("/")
    if not condition or not seed.startswith("seed") or not seed[4:].isdigit():
        raise ValueError(f"hub references look like hf:<condition>/seed<k> or hf:recommended, got {reference!r}")
    return condition, int(seed[4:])


def resolve_hub_bundle(reference: str) -> str:
    """Download (or reuse the cache of) one bundle and return its local directory."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("pip install huggingface_hub to use hf: bundle references") from exc

    repo = hub_repo_id()
    condition, seed = parse_reference(reference)
    if condition == "recommended":
        import json

        pointer = hf_hub_download(repo, RECOMMENDED_POINTER, token=os.environ.get("HF_TOKEN"))
        with open(pointer) as handle:
            value = json.load(handle)
        condition, seed = str(value["condition"]), int(value["seed"])
    folder = f"bundles/{condition}/seed{seed}"
    root = snapshot_download(repo, allow_patterns=[f"{folder}/*"], token=os.environ.get("HF_TOKEN"))
    return os.path.join(root, folder)
