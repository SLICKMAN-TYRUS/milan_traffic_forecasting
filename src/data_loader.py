"""
data_loader.py
==============
Memory-efficient loading and preprocessing of the Telecom Italia Big Data
Challenge dataset (Milan SMS/Call/Internet activity).

Strategy
--------
Each raw daily file has ~4–5 million rows (one per square × time × country).
Loading all files at once would require tens of GB of RAM.

Our approach:
  1. Read only the three columns we need (square_id, time_interval, internet).
  2. Aggregate (sum) internet traffic per (square_id, time_interval) within
     each file — collapsing country-code rows immediately.
  3. Downcast dtypes to float32 / int32 to halve memory usage.
  4. Concatenate the per-day aggregated DataFrames (tiny compared to raw).
  5. Pivot to a (T × 10000) matrix and forward-fill sparse gaps.
  6. Persist the processed matrix as a compressed Parquet file for reuse.

Column layout of raw files (tab-separated, no header):
  0  square_id     int
  1  time_interval Unix ms timestamp
  2  country_code  int
  3  sms_in        float (often NaN)
  4  sms_out       float (often NaN)
  5  call_in       float (often NaN)
  6  call_out      float (often NaN)
  7  internet      float (often NaN)  ← we want this
"""

import os
import glob
import argparse
import tracemalloc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------
RAW_COLS      = [0, 1, 7]               # square_id, time_interval, internet
COL_NAMES     = ["square_id", "time_interval", "internet"]
DTYPES_RAW    = {0: "int32", 1: "int64", 7: "float32"}


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _mem_mb() -> float:
    """Current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _df_mem_mb(df: pd.DataFrame) -> float:
    """Memory used by a DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------

def load_single_file(filepath: str) -> pd.DataFrame:
    """
    Load one raw daily file and return an aggregated DataFrame.

    Returns
    -------
    pd.DataFrame with columns [square_id, time_interval, internet]
    and one row per (square_id, time_interval) pair.
    The internet column is the *sum* across all country codes.
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        usecols=RAW_COLS,
        names=COL_NAMES,
        dtype=DTYPES_RAW,
        na_values=["", " "],
    )

    # Fill missing internet values with 0 before aggregation
    df["internet"] = df["internet"].fillna(0.0)

    # Aggregate: sum internet across country codes for each (cell, timestamp)
    agg = (
        df.groupby(["square_id", "time_interval"], sort=False)["internet"]
        .sum()
        .reset_index()
    )

    # Keep float32 to save memory
    agg["internet"] = agg["internet"].astype("float32")
    agg["square_id"] = agg["square_id"].astype("int32")

    return agg


# ---------------------------------------------------------------------------
# Full dataset loader
# ---------------------------------------------------------------------------

def load_all_files(
    data_dir: str,
    pattern: str = "sms-call-internet-mi-*.txt",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Iteratively load and aggregate all daily files in data_dir.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw .txt files.
    pattern : str
        Glob pattern for file matching.
    verbose : bool
        Print progress and memory info.

    Returns
    -------
    pd.DataFrame with columns [square_id, time_interval, internet],
    sorted by time_interval then square_id.
    """
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{data_dir}'"
        )

    if verbose:
        print(f"Found {len(files)} files.")
        print(f"Memory before loading: {_mem_mb():.1f} MB")

    chunks = []
    for i, fpath in enumerate(files, 1):
        t0 = time.time()
        chunk = load_single_file(fpath)
        elapsed = time.time() - t0
        if verbose:
            print(
                f"  [{i:02d}/{len(files)}] {Path(fpath).name}  "
                f"rows={len(chunk):,}  "
                f"mem={_df_mem_mb(chunk):.1f}MB  "
                f"time={elapsed:.1f}s"
            )
        chunks.append(chunk)

    combined = pd.concat(chunks, ignore_index=True)
    del chunks

    if verbose:
        print(f"\nCombined shape (raw agg): {combined.shape}")
        print(f"Memory after loading:  {_mem_mb():.1f} MB")

    return combined


# ---------------------------------------------------------------------------
# Pivot & clean
# ---------------------------------------------------------------------------

def build_traffic_matrix(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pivot aggregated long-form data into a wide matrix.

    Returns
    -------
    pd.DataFrame of shape (T, 10000) where:
      - Index   : pd.DatetimeIndex (UTC, 10-minute frequency)
      - Columns : square_id (int)
    """
    if verbose:
        print("\nPivoting to (time × square) matrix …")

    mem_before = _mem_mb()

    # Convert ms timestamp to datetime
    df["datetime"] = pd.to_datetime(df["time_interval"], unit="ms", utc=True)

    pivot = df.pivot_table(
        index="datetime",
        columns="square_id",
        values="internet",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Ensure float32
    pivot = pivot.astype("float32")

    # Sort index
    pivot = pivot.sort_index()

    # Forward-fill any remaining gaps (max 1 interval = 10 min)
    pivot = pivot.ffill(limit=1).fillna(0.0)

    mem_after = _mem_mb()

    if verbose:
        print(f"  Matrix shape : {pivot.shape}")
        print(f"  Date range   : {pivot.index[0]} → {pivot.index[-1]}")
        print(f"  Memory before pivot: {mem_before:.1f} MB")
        print(f"  Memory after pivot : {mem_after:.1f} MB")
        n_missing = pivot.isnull().sum().sum()
        print(f"  Remaining NaN: {n_missing}")

    return pivot


# ---------------------------------------------------------------------------
# Memory optimisation demo (for report Section 1)
# ---------------------------------------------------------------------------

def memory_optimisation_demo(filepath: str) -> dict:
    """
    Load one file with and without optimisations and return memory stats.
    Used to produce the before/after evidence required by Section 1.

    Parameters
    ----------
    filepath : str  Path to a single raw .txt file.

    Returns
    -------
    dict with keys: naive_mb, optimised_mb, ratio
    """
    # --- NAIVE (no optimisation) ---
    tracemalloc.start()
    df_naive = pd.read_csv(filepath, sep="\t", header=None)
    naive_current, naive_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    naive_mb = naive_peak / 1024 ** 2

    # --- OPTIMISED ---
    tracemalloc.start()
    df_opt = load_single_file(filepath)
    opt_current, opt_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    opt_mb = opt_peak / 1024 ** 2

    ratio = naive_mb / opt_mb if opt_mb > 0 else float("inf")

    print("=" * 50)
    print("Memory Optimisation Demo")
    print("=" * 50)
    print(f"  Naive   peak memory : {naive_mb:.1f} MB  (shape {df_naive.shape})")
    print(f"  Optimised peak memory: {opt_mb:.1f} MB  (shape {df_opt.shape})")
    print(f"  Reduction factor    : {ratio:.1f}×")
    print("=" * 50)

    return {"naive_mb": naive_mb, "optimised_mb": opt_mb, "ratio": ratio}


# ---------------------------------------------------------------------------
# Save / load processed matrix
# ---------------------------------------------------------------------------

def save_processed(matrix: pd.DataFrame, output_path: str) -> None:
    """Save pivot matrix as compressed Parquet."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(output_path, compression="snappy")
    size_mb = os.path.getsize(output_path) / 1024 ** 2
    print(f"Saved processed matrix → {output_path}  ({size_mb:.1f} MB on disk)")


def load_processed(path: str) -> pd.DataFrame:
    """Load previously saved Parquet matrix."""
    df = pd.read_parquet(path)
    df = df.astype("float32")
    print(f"Loaded processed matrix: {df.shape}  from {path}")
    return df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load and preprocess Milan telecom dataset."
    )
    parser.add_argument("--data_dir",   required=True, help="Directory with raw .txt files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed Parquet")
    parser.add_argument("--demo_file",  default=None,  help="Single file for memory demo")
    args = parser.parse_args()

    if args.demo_file:
        memory_optimisation_demo(args.demo_file)

    df_long   = load_all_files(args.data_dir)
    matrix    = build_traffic_matrix(df_long)
    out_path  = os.path.join(args.output_dir, "traffic_matrix.parquet")
    save_processed(matrix, out_path)


if __name__ == "__main__":
    main()
