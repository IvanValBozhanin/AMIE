# CONTRACT (Module)
# Purpose:
#   Adaptive two-state Kalman filter that turns return features into trend scores and uncertainties.
# Public API:
#   - KalmanFilter(...).fit(features: pd.DataFrame) -> None
#   - KalmanFilter(...).predict(features: pd.DataFrame) -> Sequence[Signal]
#   - KalmanFilter(...).get_uncertainty(features: pd.DataFrame) -> np.ndarray
# Inputs:
#   - features: DataFrame with columns {'ts','instrument','returns'}; unsorted indices are sorted by 'ts';
#     'returns' cast to float, NaN/inf allowed and treated as missing observations.
# Outputs:
#   - predict: list of Signal(len == len(features)) with ts/instrument propagated from features, ordered by ts.
#   - get_uncertainty: np.ndarray[float] length len(features) giving sqrt of trend variance per row.
#   - Internal state: 2x1 state vector [level, trend] and 2x2 covariance persisted when predictions advance.
# Invariants:
#   - Transition F=[[1,1],[0,1]]; observation H=[0,1]; only 'returns' observed; state/covariance float64.
#   - Missing or non-finite returns trigger prediction-only step; covariance stays positive semidefinite with
#     innovation variance S clipped to >=1e-12.
#   - Adaptive Q/R variances updated via EWMA and clipped to finite bands; determinism given identical inputs.
#   - Signals at index t depend only on features rows <= t after sorting; no future leakage post-sort.
#   - Signal.score equals trend estimate; Signal.uncertainty = sqrt(max(trend_var,0)) >= 0; length preserved.
# TODO:
#   - Confirm external call sites expect auto-fit behaviour when fit() not called explicitly.

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
        # MINI-CONTRACT: KalmanFilter.fit
        # Inputs:
        #   features: DataFrame with column 'returns' (float convertible); first warmup_period rows
        #   may include NaNs/inf which are ignored when seeding state.
        # Outputs:
        #   Seeds internal state vector [[level],[trend]] using NaN-safe sums/means; covariance reset to I.
        # Invariants:
        #   - Uses min(len(features), warmup_period) observations; falls back to 0.0 if all missing.
        #   - Leaves filter marked fitted; no mutation of caller-provided DataFrame.
        # --- NaN-safe warmup slice ---
        returns = self._extract_returns(features).astype(float)
        window = min(len(returns), max(self.warmup_period, 1))
        warm = returns.iloc[:window].replace([np.inf, -np.inf], np.nan)
    
        # NaN-safe aggregates
        trend_init = float(np.nanmean(warm))        # mean return for trend
        level_init = float(np.nansum(warm))         # sum of returns for level proxy
    
        # Fallbacks if warmup is all-NaN
        if not np.isfinite(trend_init):
            trend_init = 0.0
        if not np.isfinite(level_init):
            level_init = 0.0
    
        # Seed state and covariance
        self._state = np.array([[level_init], [trend_init]], dtype=float)
        self._covariance = np.eye(2, dtype=float)
        # Optional sanity check (helps catch bad inputs early)
        assert np.isfinite(self._state).all(), "Kalman state initialized to NaN/inf"
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
        # MINI-CONTRACT: KalmanFilter._filter
        # Inputs:
        #   features: DataFrame with {'ts','instrument','returns'}; unsorted tolerated, NaNs skip updates.
        #   advance_state: bool controlling persistence of state/covariance/adaptive variances.
        #   collect_signals: bool toggling construction of Signal objects vs uncertainties only.
        # Outputs:
        #   (signals, uncertainties) where signals is list[Signal] if requested, else empty tuple; uncertainties
        #   is np.ndarray length len(features) with non-negative floats aligned to sorted rows.
        # Invariants:
        #   - Auto-calls fit() once if not previously fitted.
        #   - Each iteration performs KF predict/update with innovation-variance clipping at >=1e-12.
        #   - Non-finite returns -> prediction-only step; adaptive Q/R mean-revert toward base variances.
        #   - When advance_state=False, original state/covariance/variances remain unchanged.
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
        # MINI-CONTRACT: KalmanFilter.predict
        # Inputs:
        #   features: DataFrame with {'ts','instrument','returns'}; NaNs allowed.
        # Outputs:
        #   Sequence[Signal] length len(features), sorted by 'ts'; updates internal state and adaptive variance.
        # Invariants:
        #   - Persists state/covariance after processing; deterministic given current state and features.
        #   - Signal fields mirror feature ts/instrument; score=trend, uncertainty>=0.
        signals, _ = self._filter(features, advance_state=True, collect_signals=True)
        return signals

    def get_uncertainty(self, features: pd.DataFrame) -> np.ndarray:
        """Return uncertainties without mutating the internal filter state."""
        # MINI-CONTRACT: KalmanFilter.get_uncertainty
        # Inputs:
        #   features: DataFrame with {'ts','instrument','returns'}; processed via sorted copy.
        # Outputs:
        #   np.ndarray length len(features) with non-negative uncertainties; internal state unchanged.
        # Invariants:
        #   - Returns align with sorted features rows; repeated calls leave filter state/covariance intact.
        _, uncertainties = self._filter(features, advance_state=False, collect_signals=False)
        return uncertainties
