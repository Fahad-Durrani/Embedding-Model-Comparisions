"""Lazy, cached download of model files from the HuggingFace Hub.

The LiteRT (TFLite) Gecko models ship as large single-file artifacts (the f32
variant is ~449 MB). We do not commit them; instead they are pulled on first use
into a gitignored local cache under `config.MODELS_DIR`. `hf_hub_download` is
idempotent and content-addressed, so repeat runs resolve to the cached file
instantly with no re-download.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402


def ensure_local(repo_id: str, filename: str, subdir: str) -> str:
    """Return an absolute local path to `filename` from `repo_id`, downloading
    it into `config.MODELS_DIR/<subdir>` on first use."""
    from huggingface_hub import hf_hub_download

    local_dir = os.path.join(config.MODELS_DIR, subdir)
    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
