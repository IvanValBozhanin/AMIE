"""Filesystem-backed feature storage built on Parquet partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["FeatureStore"]


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    return series


def _schema_dict(schema: pa.Schema) -> dict[str, str]:
    return {field.name: str(field.type) for field in schema}


@dataclass
class FeatureStore:
    """Store feature DataFrames partitioned by run and calendar date."""

    base_path: Path = Path("data/features")

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        return self.base_path / run_id

    def _metadata_path(self, run_id: str) -> Path:
        return self._run_path(run_id) / "_metadata.json"

    def _load_metadata(self, run_id: str) -> dict[str, object]:
        metadata_path = self._metadata_path(run_id)
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found for run_id '{run_id}'.")
        return json.loads(metadata_path.read_text())

    def write(
        self,
        df: pd.DataFrame,
        run_id: str,
        *,
        partition_by: str = "date",
    ) -> None:
        """Persist features to Parquet partitions for a given run."""
        if "ts" not in df:
            raise ValueError("Input DataFrame must contain a 'ts' column.")

        run_path = self._run_path(run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        df = df.copy()
        df["ts"] = _ensure_datetime(df["ts"])

        if partition_by == "date":
            df["__partition_key"] = df["ts"].dt.date
        else:
            if partition_by not in df:
                raise ValueError(f"Partition column '{partition_by}' missing.")
            df["__partition_key"] = df[partition_by]

        record_table = pa.Table.from_pandas(df.drop(columns="__partition_key"))
        schema_info = _schema_dict(record_table.schema)

        for partition_value, partition_df in df.groupby("__partition_key"):
            key_str = str(partition_value)
            partition_path = run_path / f"{key_str}.parquet"
            table = pa.Table.from_pandas(
                partition_df.drop(columns="__partition_key"),
                schema=record_table.schema,
            )
            pq.write_table(table, partition_path)

        metadata = {
            "schema": schema_info,
            "row_count": int(len(df)),
            "created_at": datetime.now(UTC).isoformat(),
            "partition_field": partition_by,
        }
        self._metadata_path(run_id).write_text(json.dumps(metadata, indent=2))

    def _iter_partitions(
        self,
        run_id: str,
    ) -> Iterable[Path]:
        run_path = self._run_path(run_id)
        if not run_path.exists():
            return []
        return sorted(
            path
            for path in run_path.iterdir()
            if path.suffix == ".parquet"
        )

    def read(
        self,
        run_id: str,
        *,
        start_date: Optional[str | datetime | pd.Timestamp] = None,
        end_date: Optional[str | datetime | pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Load feature partitions within an optional date range."""
        metadata = self._load_metadata(run_id)
        expected_schema = metadata["schema"]

        def _as_date(value: Optional[str | datetime | pd.Timestamp]) -> Optional[pd.Timestamp]:
            if value is None:
                return None
            return pd.to_datetime(value).normalize()

        start = _as_date(start_date)
        end = _as_date(end_date)

        frames: list[pd.DataFrame] = []
        for partition_path in self._iter_partitions(run_id):
            partition_date = pd.to_datetime(partition_path.stem).normalize()
            if start is not None and partition_date < start:
                continue
            if end is not None and partition_date > end:
                continue

            table = pq.read_table(partition_path)
            schema_dict = _schema_dict(table.schema)
            if schema_dict != expected_schema:
                raise ValueError(
                    f"Schema mismatch for '{partition_path.name}'. "
                    f"Expected {expected_schema}, found {schema_dict}."
                )
            frames.append(table.to_pandas())

        # TODO(mypy): has no attribute "keys"
        if not frames:
            return pd.DataFrame(columns=list(expected_schema.keys()))

        result = pd.concat(frames, ignore_index=True)
        if "ts" in result:
            result.sort_values("ts", inplace=True)
            result.reset_index(drop=True, inplace=True)
        return result

    def list_runs(self) -> List[str]:
        """Return all available run identifiers."""
        return sorted(
            path.name
            for path in self.base_path.iterdir()
            if path.is_dir()
        )
