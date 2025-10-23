from __future__ import annotations

import inspect

import pytest

from amie.models.base import BaseModel
from amie.models.kalman import KalmanFilter
from amie.models.registry import MODEL_REGISTRY, get_model


def test_get_model_returns_kalman_filter() -> None:
    cfg = {"instrument": "ETH-USD", "process_noise": 1e-4, "observation_noise": 1e-3, "warmup_period": 5}
    model = get_model("kalman", cfg)
    assert isinstance(model, KalmanFilter)
    assert model.instrument == "ETH-USD"


def test_get_model_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        get_model("unknown_model")


def test_registry_models_implement_predict() -> None:
    for name, constructor in MODEL_REGISTRY.items():
        model = constructor({})
        assert isinstance(model, BaseModel)
        assert callable(getattr(model, "predict", None))
        signature = inspect.signature(model.predict)
        assert "features" in signature.parameters
