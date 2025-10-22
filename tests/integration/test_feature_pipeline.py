from __future__ import annotations

from pathlib import Path

import pandas.testing as pdt
# import pytest
from omegaconf import OmegaConf

from amie.features.store import FeatureStore
from scripts.compute_features import run_pipeline


def _build_cfg(tmp_path: Path, *, run_id: str = "test-run", seed: int = 7) -> OmegaConf:
    return OmegaConf.create(
        {
            "run": {"id": run_id},
            "data": {"seed": seed, "num_ticks": 100},
            "features": {"window_size": 10},
            "output": {"base_path": str(tmp_path / "features")},
        }
    )


def test_pipeline_writes_and_validates_features(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    result = run_pipeline(cfg)

    store = FeatureStore(base_path=Path(cfg.output.base_path))
    features = store.read(cfg.run.id)

    assert not features.empty
    warmup = cfg.features.window_size
    assert features.iloc[warmup:].notna().all().all()
    assert (tmp_path / "features" / cfg.run.id).exists()
    assert result["stats"]["row_count"] == len(features)


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, run_id="det-run", seed=42)
    run_pipeline(cfg)
    store = FeatureStore(base_path=Path(cfg.output.base_path))
    first = store.read(cfg.run.id).copy()

    run_pipeline(cfg)
    second = store.read(cfg.run.id)

    pdt.assert_frame_equal(first, second)
