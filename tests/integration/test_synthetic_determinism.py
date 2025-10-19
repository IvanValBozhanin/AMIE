import hashlib
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401

from amie.data.sources.synthetic_lob import SyntheticLOBGenerator

GOLDEN_DIR = Path("tests/fixtures/golden")
PARQUET_PATH = GOLDEN_DIR / "synthetic_data_seed42.parquet"
HASH_PATH = GOLDEN_DIR / "synthetic_data_seed42.sha256"


def load_golden_hash() -> str:
    return HASH_PATH.read_text().strip()


def compute_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_synthetic_generator_matches_golden(tmp_path):
    generator = SyntheticLOBGenerator(seed=42)
    df = generator.to_dataframe(num_ticks=1_000)
    output = tmp_path / "synthetic.parquet"
    df.to_parquet(output, index=False)

    runtime_hash = compute_hash(output)
    golden_hash = load_golden_hash()

    assert runtime_hash == golden_hash, (
        "SyntheticLOBGenerator output diverged from golden fixture.\n"
        "Re-run scripts/generate_golden_files.py if the change is intentional."
    )
