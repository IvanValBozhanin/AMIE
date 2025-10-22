"""Kalman filter based signal model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from amie.core.types import Signal
from amie.models.base import BaseModel
from amie.utils.profiling import profile

__all__ = ["KalmanFilter"]


def _to_datetime(value):
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


@dataclass
class KalmanFilter(BaseModel):
    """Two-state Kalman filter tracking price level and trend."""

    instrument: str
    process_noise: float = 1e-3
    observation_noise: float = 1e-2
    warmup_period: int = 10
    model_version: str = "kalman_v0.1"


    # NEW: simple innovation-based covariance inflation
    adaptive_inflation: bool = True
    inflation_threshold: float = 3.0   # trigger if (residual^2 / S) > threshold
    inflation_strength: float = 0.25   # how strongly to inflate covariance
    inflation_cap: float = 5.0         # safety cap on single-step inflation

    def __post_init__(self) -> None:
        self._state = np.zeros((2, 1), dtype=float)
        self._covariance = np.eye(2, dtype=float)
        self._fitted = False

        self._F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        self._H = np.array([[0.0, 1.0]], dtype=float)

        # Base noises as *variances* (inputs are stddevs)
        self._Q_level = 1e-12
        self._Q_base_trend = float(self.process_noise) ** 2       # trend process noise (Q22 base)
        self._R_var_base  = float(self.observation_noise) ** 2    # observation variance (R base)

        # EWMAs that we can adapt per step; we only persist when advance_state=True
        self._R_var_ewma  = self._R_var_base
        self._Q_trend_ewma = self._Q_base_trend
        

    def fit(self, features: pd.DataFrame) -> None:
        """Initialise the latent state using an initial slice of returns."""
        returns = self._extract_returns(features)
        if returns.empty:
            raise ValueError("Cannot fit KalmanFilter without returns data.")

        window = min(len(returns), max(self.warmup_period, 1))
        warmup_returns = returns.iloc[:window]

        # --- seed state from warmup ---
        warm = warmup_returns.to_numpy(dtype=float)

        # initial state: price level ≈ mean cumulative return, trend ≈ mean return
        self._state = np.array([[float(np.cumsum(warm).mean())],
                                [float(warm.mean())]], dtype=float)

        # initial trend variance from warmup volatility; enforce a generous prior
        if warm.size > 1:
            trend_var0 = float(np.var(warm, ddof=1))
        else:
            trend_var0 = 0.0

        # Do not start over-confident: at least account for process and observation noise.
        trend_var0 = max(trend_var0, self._Q_base_trend)
        trend_var0 = max(trend_var0, self._R_var_base)
        trend_var0 = max(trend_var0, 1e-10)

        self._covariance = np.array([[1e-10, 0.0],
                                    [0.0,    trend_var0]], dtype=float)

        self._fitted = True

    def _extract_returns(self, features: pd.DataFrame) -> pd.Series:
        if "returns" not in features:
            raise ValueError("Features DataFrame must contain a 'returns' column.")
        return features["returns"].astype(float)

    def _prepare_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if not {"ts", "instrument", "returns"}.issubset(features.columns):
            missing = {"ts", "instrument", "returns"}.difference(features.columns)
            raise ValueError(f"Features missing required columns: {missing}")
        df = features.copy()
        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _filter(
        self,
        features: pd.DataFrame,
        *,
        advance_state: bool,
        collect_signals: bool,
    ) -> tuple[Sequence[Signal], np.ndarray]:
        df = self._prepare_frame(features)
        if not self._fitted:
            # Auto-fit if not explicitly fitted yet.
            self.fit(df)

        state = self._state.copy()
        covariance = self._covariance.copy()

        signals: list[Signal] = []
        uncertainties: list[float] = []

        # local (non-mutating) copies for preview-vs-predict parity
        R_ewma  = float(getattr(self, "_R_var_ewma",  self._R_var_base))
        Q_ewma  = float(getattr(self, "_Q_trend_ewma", self._Q_base_trend))

        alpha_R = 0.15  # EWMA speed for observation variance
        beta_Q = 0.35  # EWMA speed for process noise of trend
        # Clip bands keep behavior sane
        R_min, R_max = 0.25 * self._R_var_base, 25.0 * self._R_var_base
        Q_min, Q_max = 0.50 * self._Q_base_trend, 1000.0 * self._Q_base_trend

        for row in df.itertuples(index=False):
            observation = float(getattr(row, "returns"))

            # --- Prediction with *adaptive* Q (trend)
            Q_eff = float(np.clip(Q_ewma, Q_min, Q_max))
            R_eff = float(np.clip(R_ewma, R_min, R_max))
            Q_mat = np.array(
                [[self._Q_level, 0.0], [0.0, Q_eff]],
                dtype=float,
            )
            state_pred = self._F @ state
            covariance_pred = self._F @ covariance @ self._F.T + Q_mat

            if np.isfinite(observation):
                # Innovation
                residual = observation - float((self._H @ state_pred).item())
                hp = float((self._H @ covariance_pred @ self._H.T).item())
                S = max(hp + R_eff, 1e-12)

                z2 = (residual * residual) / S  # normalized squared innovation
                if self.adaptive_inflation and z2 > self.inflation_threshold:
                    # High surprise: bump covariance to react faster.
                    overshoot = z2 - self.inflation_threshold
                    scale = 1.0 + self.inflation_strength * overshoot
                    scale = float(np.clip(scale, 1.0, self.inflation_cap))
                    covariance_pred *= scale
                    hp = float((self._H @ covariance_pred @ self._H.T).item())
                    S = max(hp + R_eff, 1e-12)

                # Kalman update
                kalman_gain = (covariance_pred @ self._H.T) / S
                state = state_pred + kalman_gain * residual

                I_2 = np.eye(2)
                I_KH = I_2 - kalman_gain @ self._H
                covariance = I_KH @ covariance_pred @ I_KH.T + R_eff * (kalman_gain @ kalman_gain.T)

                residual_sq = residual * residual
                R_ewma = (1.0 - alpha_R) * R_ewma + alpha_R * residual_sq
                # Encourage larger process noise when innovations grow.
                target_Q = self._Q_base_trend + residual_sq
                Q_ewma = (1.0 - beta_Q) * Q_ewma + beta_Q * target_Q

            else:
                state = state_pred
                covariance = covariance_pred
                # Without an observation we decay the adaptive terms gently.
                R_ewma = (1.0 - alpha_R) * R_ewma + alpha_R * self._R_var_base
                Q_ewma = (1.0 - beta_Q) * Q_ewma + beta_Q * self._Q_base_trend

            # Output uncertainty = sqrt(var of trend)
            trend_var = float(covariance[1, 1])
            uncertainties.append(float(np.sqrt(max(trend_var, 0.0))))

            if collect_signals:
                signals.append(
                    Signal(
                        ts=_to_datetime(getattr(row, "ts")),
                        instrument=getattr(row, "instrument"),
                        score=float(state[1, 0]),
                        uncertainty=uncertainties[-1],
                        model_version=self.model_version,
                    )
                )

        # persist ONLY if we advanced the real state
        if advance_state:
            self._state = state
            self._covariance = covariance
            self._R_var_ewma = R_ewma
            self._Q_trend_ewma = Q_ewma

        return signals, np.asarray(uncertainties, dtype=float)

    @profile("KalmanFilter.predict")
    def predict(self, features: pd.DataFrame) -> Sequence[Signal]:
        """Run the Kalman filter over features and produce signals."""
        signals, _ = self._filter(features, advance_state=True, collect_signals=True)
        return signals

    def get_uncertainty(self, features: pd.DataFrame) -> np.ndarray:
        """Return uncertainties without mutating the internal filter state."""
        _, uncertainties = self._filter(features, advance_state=False, collect_signals=False)
        return uncertainties
