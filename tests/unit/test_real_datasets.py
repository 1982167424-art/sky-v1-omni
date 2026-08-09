"""Unit tests for sky_v1.data.real_datasets + the 4 download scripts (spec §11).

These tests NEVER do real network downloads:
  - conftest.py sets SKY_V1_TEST_MODE=1, which short-circuits
    RealDatasetLoader._try_load_* and forces the toy fallback path.
  - Even without that flag, the lazy ``import datasets`` / ``import modelscope``
    inside the loader raises ImportError in this env (not in requirements.txt),
    which the loader catches and converts to toy samples.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from sky_v1.data.real_datasets import RealDatasetLoader


def test_real_dataset_loader_init():
    """RealDatasetLoader can be instantiated and exposes its API surface."""
    loader = RealDatasetLoader()
    assert isinstance(loader, RealDatasetLoader)
    assert loader.SOURCES == ["hf", "aliyun", "modelscope"]
    assert callable(loader.load)
    assert callable(loader.list_available)


def test_list_available_returns_dict():
    """list_available() returns a non-empty dict covering the expected categories."""
    loader = RealDatasetLoader()
    available = loader.list_available()
    assert isinstance(available, dict)
    assert available, "list_available() must not be empty"
    for key in ("pretrain", "sft", "preference", "modal_image", "modal_audio",
                "modal_video", "modal_3d", "sources"):
        assert key in available, f"missing category: {key}"
        assert len(available[key]) > 0, f"category {key} has no entries"


def test_load_fallback_to_toy():
    """When real download is unavailable, load() must NOT crash and returns toy."""
    loader = RealDatasetLoader()
    # Force fallback explicitly via an unknown source.
    samples = loader.load("redpajama", split="train", source="bogus_source", max_samples=4)
    assert isinstance(samples, list)
    assert len(samples) > 0, "fallback must produce at least one sample"
    # Even through the legit 'hf' source path, SKY_V1_TEST_MODE / missing
    # `datasets` lib must steer us to the toy path.
    samples_hf = loader.load("alpaca", split="train", source="hf", max_samples=3)
    assert isinstance(samples_hf, list)
    assert len(samples_hf) == 3
    for s in samples_hf:
        assert isinstance(s, dict)
        assert "text" in s
        assert s.get("toy") is True


def test_download_scripts_importable():
    """All 4 download scripts must be importable and expose ``main()``."""
    module_names = [
        "scripts.data.download_pretrain_corpus",
        "scripts.data.download_sft_dataset",
        "scripts.data.download_modal_datasets",
        "scripts.data.download_preference",
    ]
    for name in module_names:
        mod = importlib.import_module(name)
        assert hasattr(mod, "main"), f"{name} missing main()"
        assert callable(mod.main), f"{name}.main is not callable"
