#!/usr/bin/env python3
"""Remove legacy columns from every parquet in a directory tree.

Usage (default paths are the ones used in the pipeline):
    python cleanup_columns.py            # uses hard‑coded PARQUET_DIR
    python cleanup_columns.py /path/to/dataset

Columns dropped if present:
    • box_2d_coords
    • grasp_points

The script rewrites the parquet *in‑place* using pandas, preserving all
other columns and dtypes.
"""

import os
import sys
import pandas as pd
from tqdm.auto import tqdm

# ───────────── Configuration ─────────────
DEFAULT_PARQUET_DIR = os.path.expanduser(
    "~/phosphobot/recordings/lerobot_v2.1/bounding-box-test1/data/chunk-000"
)

COLUMNS_TO_DROP = ["box_2d_coords", "grasp_points"]

# ───────────── Main logic ─────────────

def strip_columns(parquet_path: str):
    """Drop columns if they exist; return True if file was modified."""
    df = pd.read_parquet(parquet_path)
    present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if not present:
        return False  # nothing to do
    df = df.drop(columns=present)
    df.to_parquet(parquet_path)
    return True


def run(parquet_dir: str):
    parquet_files = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet")
    ]
    if not parquet_files:
        print("No parquet files found in", parquet_dir)
        return

    changed = 0
    for path in tqdm(parquet_files, desc="cleaning parquets"):
        if strip_columns(path):
            changed += 1
            print("🧹", os.path.basename(path), "→ columns dropped")
    print(f"✅ Done. {changed}/{len(parquet_files)} parquets modified.")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PARQUET_DIR
    run(target_dir)
