"""Simple parquet-backed tick replay utility."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from amie.core.constants import Side
from amie.core.types import Tick

__all__ = ["DataReplay"]


class DataReplay:
    """Replay `Tick` records from a parquet dataset for deterministic testing."""

    def __init__(self, filepath: str | Path) -> None:
        self._path = Path(filepath)
        self._data = (
            pd.read_parquet(self._path)
            .sort_values("ts")
            .reset_index(drop=True)
        )

    def replay(self, start_idx: int | None = None, end_idx: int | None = None) -> Iterator[Tick]:
        """Yield ticks in timestamp order, optionally slicing via start/end indices."""
        df = self._data
        if start_idx is not None or end_idx is not None:
            df = df.iloc[start_idx:end_idx]
        for row in df.itertuples(index=False):
            ts = row.ts.to_pydatetime() if hasattr(row.ts, "to_pydatetime") else row.ts
            yield Tick(
                ts=ts,
                instrument=row.instrument,
                price=float(row.price),
                qty=float(row.qty),
                side=Side(row.side),
            )
