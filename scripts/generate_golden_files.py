"""Generate deterministic golden data artifacts for synthetic streams."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from amie.data.sources.synthetic_lob import SyntheticLOBGenerator

GOLDEN_DIR = Path("tests/fixtures/golden")
PARQUET_PATH = GOLDEN_DIR / "synthetic_data_seed42.parquet"
HASH_PATH = GOLDEN_DIR / "synthetic_data_seed42.sha256"


def generate() -> pd.DataFrame:
    generator = SyntheticLOBGenerator(seed=42)
    return generator.to_dataframe(num_ticks=1_000)


def compute_hash(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_parquet(PARQUET_PATH, index=False)
    parquet_bytes = PARQUET_PATH.read_bytes()
    hash_value = compute_hash(parquet_bytes)
    HASH_PATH.write_text(hash_value)
    print(f"Generated {PARQUET_PATH} and {HASH_PATH}")


if __name__ == "__main__":
    main()
