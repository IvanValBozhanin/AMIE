"""Model abstractions for AMIE predictive components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd

from amie.core.types import Signal

__all__ = ["BaseModel"]


class BaseModel(ABC):
    """Abstract base class that defines the modelling contract."""

    model_version: str = "base"

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> None:
        """Adjust internal parameters using the provided features."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> Sequence[Signal]:
        """Generate directional signals from the supplied features."""

    @abstractmethod
    def get_uncertainty(self, features: pd.DataFrame) -> np.ndarray:
        """Return model uncertainty scores aligned with the input features."""
