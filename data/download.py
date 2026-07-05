"""
Phase 1, Day 1 — fetch the raw Adult Census Income dataset (UCI).

Run standalone: python -m data.download
Idempotent: skips the download if the file already exists with a plausible size.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config import RAW_DATA_PATH, COLUMN_NAMES_PATH

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"

# adult.data is ~3.97 MB (32,561 rows). Anything much smaller indicates a
# truncated download (this bit us once during setup) rather than a real file.
MIN_EXPECTED_BYTES = 3_900_000


def _fetch(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)


def download():
    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size >= MIN_EXPECTED_BYTES:
        print(f"{RAW_DATA_PATH} already present, skipping download.")
    else:
        print(f"Downloading {DATA_URL} -> {RAW_DATA_PATH}")
        _fetch(DATA_URL, RAW_DATA_PATH)
        size = RAW_DATA_PATH.stat().st_size
        if size < MIN_EXPECTED_BYTES:
            raise RuntimeError(
                f"Downloaded file is only {size} bytes, expected >= "
                f"{MIN_EXPECTED_BYTES}. Download was likely truncated."
            )
        print(f"Saved {size} bytes.")

    if not COLUMN_NAMES_PATH.exists():
        print(f"Downloading {NAMES_URL} -> {COLUMN_NAMES_PATH}")
        _fetch(NAMES_URL, COLUMN_NAMES_PATH)


if __name__ == "__main__":
    download()
