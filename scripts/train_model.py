"""Train a model on features and generate predictions with diagnostics."""

from __future__ import annotations

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from structlog.contextvars import bind_contextvars

from amie.features.store import FeatureStore
from amie.models.registry import get_model
from amie.utils.logging import get_logger

logger = get_logger()


def _resolve_base_path(path_str: str) -> Path:
    """Convert config path to absolute path."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    try:
        root = Path(get_original_cwd())
    except Exception:
        root = Path.cwd()
    return root / path


def _compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    uncertainty: pd.Series,
) -> Dict[str, float]:
    """Calculate prediction quality metrics."""
    # Filter out NaN values
    mask = actual.notna() & predicted.notna() & uncertainty.notna()
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    uncertainty_clean = uncertainty[mask]
    
    if len(actual_clean) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "mean_uncertainty": np.nan,
            "median_uncertainty": np.nan,
            "max_uncertainty": np.nan,
            "n_predictions": 0,
        }
    
    # Prediction errors
    errors = actual_clean - predicted_clean
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    
    # Uncertainty statistics
    mean_unc = float(uncertainty_clean.mean())
    median_unc = float(uncertainty_clean.median())
    max_unc = float(uncertainty_clean.max())
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mean_uncertainty": mean_unc,
        "median_uncertainty": median_unc,
        "max_uncertainty": max_unc,
        "n_predictions": len(actual_clean),
    }


def run_training(cfg: DictConfig) -> Dict[str, object]:
    """Execute the model training pipeline."""
    start = time.perf_counter()
    
    # Extract configuration
    run_id = cfg.get("run", {}).get("id") or cfg.get("run", {}).get("run_id")
    if not run_id:
        run_id = datetime.utcnow().strftime("train-%Y%m%d%H%M%S")
    
    bind_contextvars(run_id=run_id)
    
    # Paths
    features_path = _resolve_base_path(
        cfg.get("paths", {}).get("data_dir", "data/features")
    )
    results_path = _resolve_base_path(
        cfg.get("paths", {}).get("results_dir", "results")
    )
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    train_split = float(cfg.get("training", {}).get("train_split", 0.8))
    feature_run_id = cfg.get("training", {}).get("feature_run_id")
    
    logger.info(
        "training_start",
        run_id=run_id,
        train_split=train_split,
        feature_run_id=feature_run_id,
    )
    
    # Load features
    store = FeatureStore(base_path=features_path)
    
    if not feature_run_id:
        # Use the most recent run
        available_runs = store.list_runs()
        if not available_runs:
            raise ValueError(f"No feature runs found in {features_path}")
        feature_run_id = available_runs[-1]
        logger.info("auto_selected_feature_run", feature_run_id=feature_run_id)
    
    features = store.read(feature_run_id)
    logger.info("features_loaded", row_count=len(features), feature_run_id=feature_run_id)
    
    if len(features) < 10:
        raise ValueError(f"Insufficient data: only {len(features)} rows")
    
    # Split data
    split_idx = int(len(features) * train_split)
    train_df = features.iloc[:split_idx].copy()
    val_df = features.iloc[split_idx:].copy()
    
    logger.info(
        "data_split",
        train_rows=len(train_df),
        val_rows=len(val_df),
        split_index=split_idx,
    )
    
    # Load and train model
    model_name = cfg.get("model", {}).get("name", "kalman")
    model = get_model(model_name, cfg.get("model", {}))
    
    logger.info("model_loaded", model_name=model_name, model_version=model.model_version)
    
    # Fit on training data
    fit_start = time.perf_counter()
    model.fit(train_df)
    fit_duration = time.perf_counter() - fit_start
    logger.info("model_fitted", duration_s=fit_duration)
    
    # Predict on FULL dataset to maintain Kalman filter state continuity
    # Then extract only validation predictions
    pred_start = time.perf_counter()
    all_signals = model.predict(features)
    pred_duration = time.perf_counter() - pred_start
    
    # Extract validation signals (after split_idx)
    val_signals = all_signals[split_idx:]
    
    logger.info(
        "predictions_generated",
        n_signals=len(val_signals),
        duration_s=pred_duration,
    )
    
    # Convert signals to DataFrame
    predictions_df = pd.DataFrame([
        {
            "ts": s.ts,
            "instrument": s.instrument,
            "predicted_return": s.score,
            "uncertainty": s.uncertainty,
        }
        for s in val_signals
    ])
    
    # Merge with actual returns for metrics
    val_with_pred = val_df[["ts", "instrument", "returns"]].merge(
        predictions_df,
        on=["ts", "instrument"],
        how="inner",
    )
    
    # Compute metrics
    metrics = _compute_metrics(
        actual=val_with_pred["returns"],
        predicted=val_with_pred["predicted_return"],
        uncertainty=val_with_pred["uncertainty"],
    )
    
    logger.info("validation_metrics", **metrics)
    
    # Save model state
    model_path = results_path / f"{run_id}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("model_saved", path=str(model_path))
    
    # Save predictions
    predictions_path = results_path / f"{run_id}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info("predictions_saved", path=str(predictions_path))
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "model_name": model_name,
        "model_version": model.model_version,
        "feature_run_id": feature_run_id,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_split": train_split,
        "metrics": metrics,
        "fit_duration_s": fit_duration,
        "pred_duration_s": pred_duration,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    metadata_path = results_path / f"{run_id}_metadata.json"
    import json
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("metadata_saved", path=str(metadata_path))
    
    duration = time.perf_counter() - start
    logger.info("training_complete", run_id=run_id, duration_s=duration)
    
    return {
        "run_id": run_id,
        "model": model,
        "predictions": predictions_df,
        "metrics": metrics,
        "duration": duration,
    }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
    