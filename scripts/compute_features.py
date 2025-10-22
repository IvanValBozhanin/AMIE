"""End-to-end feature computation pipeline orchestrated with Hydra."""

from __future__ import annotations

# import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict#, Iterable

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig#, OmegaConf
from structlog.contextvars import bind_contextvars

from amie.data.sources.synthetic_lob import SyntheticLOBGenerator
from amie.data.validation import FeatureVectorSchema
from amie.features.store import FeatureStore
from amie.features.transforms import FeatureComputer
from amie.utils.logging import get_logger

logger = get_logger()


def _resolve_base_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    try:
        root = Path(get_original_cwd())
    except Exception:  # pragma: no cover - fallback when not under Hydra
        root = Path.cwd()
    return root / path


def _feature_stats(df) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for column in ("returns", "ewma_volatility", "z_score", "spread", "imbalance"):
        if column not in df:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        stats[f"{column}_min"] = float(series.min())
        stats[f"{column}_max"] = float(series.max())
        stats[f"{column}_mean"] = float(series.mean())
    return stats


def run_pipeline(cfg: DictConfig) -> Dict[str, object]:
    """Execute the synthetic data → feature pipeline."""
    start = time.perf_counter()

    run_id = cfg.get("run", {}).get("id") or cfg.get("run", {}).get("run_id")
    if not run_id:
        run_id = datetime.utcnow().strftime("run-%Y%m%d%H%M%S")

    data_cfg = cfg.get("data", {})
    seed = int(data_cfg.get("seed", 42))
    num_ticks = int(data_cfg.get("num_ticks", 1_000))

    features_cfg = cfg.get("features", {})
    window_size = int(features_cfg.get("window_size", 20))

    output_cfg = cfg.get("output", {})
    base_path = _resolve_base_path(output_cfg.get("base_path", "data/features"))

    bind_contextvars(run_id=run_id)
    logger.info(
        "pipeline_start",
        run_id=run_id,
        seed=seed,
        num_ticks=num_ticks,
        window_size=window_size,
        base_path=str(base_path),
    )

    generator = SyntheticLOBGenerator(seed=seed)
    tick_df = generator.to_dataframe(num_ticks=num_ticks)

    computer = FeatureComputer(window_size=window_size)
    features = computer.compute(tick_df)

    validation_df = (
        features.dropna().rename(columns={"ewma_volatility": "volatility"})
    )
    FeatureVectorSchema.validate(validation_df, lazy=True)

    store = FeatureStore(base_path=base_path)
    store.write(features, run_id=run_id)

    stats = {
        "row_count": len(features),
        "column_count": features.shape[1],
        **_feature_stats(features),
    }

    duration = time.perf_counter() - start
    logger.info("pipeline_complete", run_id=run_id, duration_s=duration, **stats)
    return {
        "features": features,
        "run_id": run_id,
        "base_path": base_path,
        "duration": duration,
        "stats": stats,
    }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
