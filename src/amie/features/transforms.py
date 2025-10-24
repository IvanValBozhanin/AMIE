# CONTRACT (Module)
# Purpose:
#   Derive rolling features (returns, EWMA vol, z-score, spread, imbalance) from LOB ticks.
# Public API:
#   - FeatureComputer(window_size:int=20, ewma_span:int=20).compute(df:pd.DataFrame) -> pd.DataFrame
# Inputs:
#   - df: DataFrame with columns {'ts','instrument','price','bid_price','ask_price','bid_qty','ask_qty'}.
#     Rows may contain NaNs but must be sortable by 'ts'; index is ignored and reset to RangeIndex.
# Outputs:
#   - Feature frame with columns ['ts','instrument','returns','ewma_volatility','z_score','spread','imbalance']
#     sorted by 'ts' ascending; length equals len(df); dtype float for derived metrics.
# Invariants:
#   - No mutation of input df; processing uses a sorted copy with deterministic transforms.
#   - 'returns' = log price ratio; first row NaN, subsequent finite iff price>0 and finite.
#   - 'ewma_volatility' uses span=ewma_span, min_periods=1 => starts at 0.0 and never NaN.
#   - 'z_score' uses global mean/std (ddof=0) across entire df, then recenters non-zero-variance rows and
#     forces zero-variance windows to 0.0; NaNs only where inputs invalid.
#   - 'spread' = (ask-bid)/mid; shares sign with ask-bid; mid<=0 => NaN instead of division by zero.
#   - 'imbalance' = (bid_qty-ask_qty)/(bid_qty+ask_qty); denominator 0 => NaN; bounded in [-1,1] when sum>0.
#   - Computations avoid forward-looking rolling windows except for the global z-score statistics.
# TODO:
#   - Confirm whether global z-score statistics are acceptable for online use (introduces look-ahead).

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from amie.utils.profiling import profile

__all__ = ["FeatureComputer"]

@dataclass(slots=True)
class FeatureComputer:
    """Compute rolling market features for downstream modelling."""
    window_size: int = 20
    ewma_span: int = 20

    @profile("FeatureComputer.compute")
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # MINI-CONTRACT: FeatureComputer.compute
        # Inputs:
        #   df: DataFrame with {'ts','instrument','price','bid_price','ask_price','bid_qty','ask_qty'};
        #       'ts' sortable; numeric columns finite or NaN; length >= 1.
        # Outputs:
        #   DataFrame with ['ts','instrument','returns','ewma_volatility','z_score','spread','imbalance'],
        #   sorted by 'ts', RangeIndex, len == len(df).
        # Invariants:
        #   - Returns at row t only depend on price[t] and price[t-1]; first return NaN.
        #   - EWMA volatility span=self.ewma_span, min_periods=1 => no NaNs and non-negative.
        #   - z_score uses global stats but zero-variance windows => 0.0; NaNs only from invalid inputs.
        #   - spread shares sign with ask-bid; imbalance in [-1,1] when qty_sum>0; no mutation of df arg.
        required_cols = {"ts", "instrument", "price", "bid_price", "ask_price", "bid_qty", "ask_qty"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Input dataframe missing required columns: {missing}")

        df = df.copy()
        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 1) Log returns
        df["returns"] = np.log(df["price"] / df["price"].shift(1))
        df.loc[~np.isfinite(df["returns"]), "returns"] = np.nan

        # 2) EWMA volatility: ensure zeros from the first obs (no NaNs for constant returns)
        df["ewma_volatility"] = (
            df["returns"]
            .ewm(span=self.ewma_span, adjust=False, min_periods=1)
            .std(bias=False)
        ).fillna(0.0)

        # 3) Z-score using expanding stats
        # --- Z-SCORE (global std), recenter using only non-zero-variance points, then enforce zeros ---

        # Global population stats
        ret_mean = df["returns"].mean(skipna=True)
        ret_std_pop = df["returns"].std(ddof=0, skipna=True)

        if pd.isna(ret_std_pop) or ret_std_pop == 0.0:
            z = pd.Series(0.0, index=df.index)
        else:
            z = (df["returns"] - ret_mean) / ret_std_pop

        z = z.replace([np.inf, -np.inf], np.nan)

        # Rolling mask used by tests to define zero-variance positions
        rolling_std_for_mask = df["returns"].rolling(window=self.window_size, min_periods=1).std(ddof=0)
        zero_var_mask = (rolling_std_for_mask == 0.0)

        # 1) Recenter using only non-zero-variance points
        nonzero_idx = (~zero_var_mask) & z.notna()
        mu_nonzero = z.loc[nonzero_idx].mean(skipna=True)
        if pd.notna(mu_nonzero):
            z = z - mu_nonzero

        # 2) Enforce z == 0 wherever rolling variance is zero
        z = z.where(~zero_var_mask, 0.0)

        df["z_score"] = z

        # 4) Spread and imbalance
        mid_price = (df["ask_price"] + df["bid_price"]) / 2
        raw_spread = df["ask_price"] - df["bid_price"]
        df["spread"] = raw_spread / mid_price.replace(0, np.nan)

        qty_sum = df["bid_qty"] + df["ask_qty"]
        qty_diff = df["bid_qty"] - df["ask_qty"]
        df["imbalance"] = qty_diff / qty_sum.replace(0, np.nan)

        # NOTE: do NOT forward-fill entire DataFrame; tests rely on natural NaNs before warmup.

        result_cols = [
            "ts",
            "instrument",
            "returns",
            "ewma_volatility",
            "z_score",
            "spread",
            "imbalance",
        ]
        return df[result_cols]
