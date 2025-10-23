"""Model registry and factory utilities."""

from __future__ import annotations

from typing import Callable, Dict

from omegaconf import DictConfig, OmegaConf

from .base import BaseModel
from .kalman import KalmanFilter

MODEL_REGISTRY: Dict[str, Callable[[DictConfig], BaseModel]] = {
    "kalman": lambda cfg: KalmanFilter(
        instrument=cfg.get("instrument", "BTC-USD"),
        process_noise=cfg.get("process_noise", 1e-3),
        observation_noise=cfg.get("observation_noise", 1e-2),
        warmup_period=cfg.get("warmup_period", 10),
    ),
    # "gnn": TemporalGNN,  # Future placeholder
}


def get_model(name: str, config: DictConfig | dict | None = None) -> BaseModel:
    """Instantiate a model by name using registry metadata."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    cfg = config or {}
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    constructor = MODEL_REGISTRY[name]
    model = constructor(cfg)
    if not isinstance(model, BaseModel):
        raise TypeError(f"Model '{name}' must inherit from BaseModel.")
    if not hasattr(model, "predict"):
        raise TypeError(f"Model '{name}' must implement a 'predict' method.")
    return model
