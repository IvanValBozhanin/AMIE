# CONTRACT (Module)
# Purpose:
#   Convert model-generated score/uncertainty pairs into bounded target positions for execution.
# Public API:
#   - SignalPolicy(config: Mapping | None)
#   - SignalPolicy.compute_position(signal: Signal) -> float
# Inputs:
#   - config: mapping accepting optional 'policy' key with {'threshold_multiplier': float, 'max_position_size': float}.
#   - signal: amie.core.types.Signal with fields (score:float, uncertainty:float>0, instrument:str, ...).
# Outputs:
#   - Target position float clipped to [-max_position_size, max_position_size]; zero inside neutral band.
# Invariants:
#   - threshold_multiplier>0 and max_position_size>0 validated during init; stored as floats.
#   - Decision uses only the provided Signal (no internal state, order deterministic).
#   - Neutral band defined by +/- threshold_multiplier * uncertainty; ties resolve to flat position.
#   - Scaling divides directional intent by uncertainty, limiting leverage via max_position_size.
# TODO:
#   - Confirm caller guarantees strictly positive signal.uncertainty prior to policy invocation.

"""Trading policy that converts model signals into target positions."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from amie.core.types import Signal

logger = logging.getLogger(__name__)


class SignalPolicy:
    """Policy that turns signals into position targets."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        policy_config = self._extract_policy_config(config)
        self.threshold_multiplier = float(policy_config.get("threshold_multiplier", 2.0))
        self.max_position_size = float(policy_config.get("max_position_size", 1.0))

        if self.threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be positive")
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")

    @staticmethod
    def _extract_policy_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """Return the mapping that stores policy-specific settings."""
        # MINI-CONTRACT: SignalPolicy._extract_policy_config
        # Inputs:
        #   config: optional mapping, possibly containing nested 'policy' sub-mapping.
        # Outputs:
        #   Mapping with policy parameters; returns {} when config falsy.
        # Invariants:
        #   - If config['policy'] is Mapping, returns that; otherwise returns original config.
        #   - Does not mutate input mapping.
        if not config:
            return {}
        if "policy" in config and isinstance(config["policy"], Mapping):
            return config["policy"]
        return config

    def compute_position(self, signal: Signal) -> float:
        """Return desired position for the provided signal."""
        # MINI-CONTRACT: SignalPolicy.compute_position
        # Inputs:
        #   signal: Signal with float score and strictly positive uncertainty.
        # Outputs:
        #   Position float in [-max_position_size, max_position_size]; zero when score inside neutral band.
        # Invariants:
        #   - Raises ValueError if signal.uncertainty <= 0.
        #   - Uses thresholds +/- threshold_multiplier*uncertainty; scaling is base_position/uncertainty.
        #   - Deterministic mapping: identical signal values produce identical outputs; no stateful side effects.
        if signal.uncertainty <= 0:
            raise ValueError("Signal uncertainty must be positive")

        threshold = self.threshold_multiplier * signal.uncertainty
        rationale: str

        if signal.score > threshold:
            base_position = 1.0
            rationale = (
                f"score {signal.score:.4f} exceeds long threshold {threshold:.4f}"
            )
        elif signal.score < -threshold:
            base_position = -1.0
            rationale = (
                f"score {signal.score:.4f} below short threshold {-threshold:.4f}"
            )
        else:
            base_position = 0.0
            rationale = (
                f"score {signal.score:.4f} inside neutral band ±{threshold:.4f}"
            )

        if base_position == 0.0:
            position = 0.0
            scaled_position = 0.0
        else:
            scaled_position = base_position / signal.uncertainty
            position = max(
                -self.max_position_size,
                min(self.max_position_size, scaled_position),
            )

        logger.debug(
            "SignalPolicy decision: instrument=%s score=%.4f uncertainty=%.4f "
            "base_position=%.2f scaled_position=%.4f final_position=%.4f rationale=%s",
            signal.instrument,
            signal.score,
            signal.uncertainty,
            base_position,
            scaled_position,
            position,
            rationale,
        )
        return position
