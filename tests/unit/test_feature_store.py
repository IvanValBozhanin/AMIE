from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from amie.features.store import FeatureStore


def _make_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=4, freq="h")
    return pd.DataFrame(
        {
            "ts": ts,
            "instrument": ["BTC-USD"] * 4,
            "returns": [0.0, 0.01, -0.005, 0.002],
            "ewma_volatility": [0.0, 0.1, 0.2, 0.15],
            "z_score": [0.0, 1.0, -0.5, 0.2],
            "spread": [0.0004, 0.0005, 0.00045, 0.00048],
            "imbalance": [0.0, 0.2, -0.1, 0.05],
        }
    )


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    store = FeatureStore(base_path=tmp_path / "features")
    run_id = "run-123"
    df = _make_df()

    store.write(df, run_id)
    result = store.read(run_id)

    pdt.assert_frame_equal(result, df)


def test_partition_files_created(tmp_path: Path) -> None:
    store = FeatureStore(base_path=tmp_path / "features")
    df = _make_df()
    run_id = "partition-run"

    store.write(df, run_id)
    partition_files = list((tmp_path / "features" / run_id).glob("*.parquet"))
    assert len(partition_files) == len(df["ts"].dt.date.unique())


def test_schema_mismatch_raises(tmp_path: Path) -> None:
    store = FeatureStore(base_path=tmp_path / "features")
    df = _make_df()
    run_id = "schema-run"
    store.write(df, run_id)

    run_path = tmp_path / "features" / run_id
    bad_table = pa.Table.from_pandas(df.assign(extra=1))
    pq.write_table(bad_table, run_path / "2024-01-03.parquet")

    with pytest.raises(ValueError):
        store.read(run_id)


def test_list_runs_returns_ids(tmp_path: Path) -> None:
    store = FeatureStore(base_path=tmp_path / "features")
    df = _make_df()
    store.write(df, "run-1")
    store.write(df, "run-2")
    runs = store.list_runs()
    assert runs == ["run-1", "run-2"]
