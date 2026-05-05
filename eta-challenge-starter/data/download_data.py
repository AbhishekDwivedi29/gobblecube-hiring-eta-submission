#!/usr/bin/env python
"""One-time download & cleanup of NYC TLC 2023 yellow-taxi data.

Produces:
    data/train.parquet       -- 11.5 months of 2023, ~37M trips after cleaning
    data/dev.parquet         -- last 2 weeks of 2023, ~1M trips (for local grading)
    data/sample_1M.parquet   -- 1M-row subset of train for fast iteration

The held-out Eval set (a 2024 slice) is kept by Gobblecube and never distributed.

Takes ~5 minutes on a fast connection, ~20 minutes on a slow one.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = [f"2023-{m:02d}" for m in range(1, 13)]

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"

CUTOFF = pd.Timestamp("2023-12-18")   # dev = last ~2 weeks of Dec
SAMPLE_SIZE = 1_000_000


def download_month(yyyymm: str) -> Path:
    RAW_DIR.mkdir(exist_ok=True)
    url = f"{BASE_URL}/yellow_tripdata_{yyyymm}.parquet"
    out = RAW_DIR / f"yellow_{yyyymm}.parquet"
    if out.exists():
        print(f"  cached   {out.name}")
        return out
    print(f"  fetching {url}")
    urlretrieve(url, out)
    return out


def clean_base(paths: list[Path]) -> pd.DataFrame:
    """Applies universal sanity filters and removes duplicates before splitting."""
    frames = []
    for p in paths:
        df = pd.read_parquet(
            p,
            columns=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "passenger_count",
            ],
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    duration = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()

    clean_df = pd.DataFrame({
        "pickup_zone":      df["PULocationID"].astype("int32"),
        "dropoff_zone":     df["DOLocationID"].astype("int32"),
        "requested_at":     df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "passenger_count":  df["passenger_count"].fillna(1).astype("int8"),
        "duration_seconds": duration.astype("float64"),
        "_ts":              df["tpep_pickup_datetime"],
    })

    # Apply base sanity filters
    mask = (
        (clean_df["duration_seconds"] >= 30)
        & (clean_df["duration_seconds"] <= 3 * 3600)
        & (clean_df["pickup_zone"].between(1, 265))
        & (clean_df["dropoff_zone"].between(1, 265))
        & (clean_df["_ts"].dt.year == 2023)
    )
    
    clean_df = clean_df.loc[mask].reset_index(drop=True)

    # --- NEW: Check for and remove duplicates ---
    duplicate_count = clean_df.duplicated().sum()
    print(f"  Found {duplicate_count:,} duplicate records.")
    
    if duplicate_count > 0:
        clean_df = clean_df.drop_duplicates(ignore_index=True)
        print("  Duplicates successfully removed.")

    return clean_df


def apply_speed_ratio_filter(df: pd.DataFrame, osrm_csv_path: str | Path) -> pd.DataFrame:
    """Filters records based on the ratio of actual speed to OSRM speed."""
    initial_count = len(df)
    
    # Load the OSRM matrix
    osrm_df = pd.read_csv(osrm_csv_path)
    
    # Merge OSRM distance and speed data onto our main dataframe
    merged = df.merge(
        osrm_df[["pickup_zone", "dropoff_zone", "osrm_distance_meters", "speed_kmph"]],
        on=["pickup_zone", "dropoff_zone"],
        how="left"
    )
    
    # Calculate actual speed in km/h: (meters / seconds) * 3.6
    # Note: duration_seconds is guaranteed to be >= 30 from clean_base
    actual_speed_kmph = (merged["osrm_distance_meters"] / merged["duration_seconds"]) * 3.6
    
    # Calculate ratio (replace 0.0 with NA to prevent division by zero for same-zone trips)
    speed_ratio = actual_speed_kmph / merged["speed_kmph"].replace(0.0, pd.NA)
    
    # Create masks
    ratio_mask = (speed_ratio >= 0.2) & (speed_ratio <= 1.5)
    
    # Preserve same-zone trips. OSRM gives them 0 distance/speed, causing ratio to be NA.
    # They are strictly required for the thresholds in Step 4.
    same_zone_mask = (df["pickup_zone"] == df["dropoff_zone"])
    
    final_mask = ratio_mask | same_zone_mask
    
    filtered_df = df.loc[final_mask].reset_index(drop=True)
    
    dropped_count = initial_count - len(filtered_df)
    print(f"  Dropped {dropped_count:,} records outside speed ratio 0.2 - 1.5")
    
    return filtered_df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and dev based on the CUTOFF date."""
    train = df[df["_ts"] < CUTOFF].drop(columns=["_ts"]).reset_index(drop=True)
    dev = df[df["_ts"] >= CUTOFF].drop(columns=["_ts"]).reset_index(drop=True)
    return train, dev


def apply_leak_free_thresholds(train: pd.DataFrame, dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates 99th percentile limits on train ONLY and applies to both train and dev."""
    
    # 1. Isolate same-zone trips strictly from the training set
    train_same_zone = train[train["pickup_zone"] == train["dropoff_zone"]]
    
    # 2. Calculate thresholds for specific known routes
    p99_thresholds = train_same_zone.groupby("pickup_zone")["duration_seconds"].quantile(0.99)
    
    # 3. Calculate a global fallback for unseen routes in dev
    global_p99 = train_same_zone["duration_seconds"].quantile(0.99)

    def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
        is_same_zone = df["pickup_zone"] == df["dropoff_zone"]
        
        # Map the specific thresholds to the dataframe. 
        # If a zone isn't in the train thresholds (NaN), fill it with the global fallback.
        mapped_thresholds = df["pickup_zone"].map(p99_thresholds).fillna(global_p99)
        
        # Keep row if: Not same-zone OR (same-zone AND duration <= threshold)
        keep_mask = (~is_same_zone) | (df["duration_seconds"] <= mapped_thresholds)
        
        return df.loc[keep_mask].reset_index(drop=True)

    # 4. Filter both datasets using the train-derived limits
    train_filtered = filter_outliers(train)
    dev_filtered = filter_outliers(dev)
    
    return train_filtered, dev_filtered


def main() -> None:
    print("Step 1: download monthly parquets")
    paths = [download_month(m) for m in MONTHS]

    print("\nStep 2: base clean & combine")
    df = clean_base(paths)
    print(f"  base cleaned: {len(df):,} trips")

    print("\nStep 3: filter using OSRM speed ratio")
    # Make sure 'osrm_matrix_with_speed.csv' is in your working directory or adjust the path below
    osrm_csv = "data/osrm_matrix_with_speed.csv" 
    df = apply_speed_ratio_filter(df, osrm_csv)
    print(f"  records remaining: {len(df):,}")

    print("\nStep 4: train/dev split (to prevent data leakage)")
    train, dev = split(df)

    print("\nStep 5: calculate & apply 99th percentile limits (same-zone)")
    train, dev = apply_leak_free_thresholds(train, dev)
    
    print("\nStep 6: save final outputs")
    train.to_parquet(DATA_DIR / "train.parquet", index=False)
    dev.to_parquet(DATA_DIR / "dev.parquet", index=False)
    print(f"  train.parquet: {len(train):,} rows")
    print(f"  dev.parquet:   {len(dev):,} rows")

    print("\nStep 7: 1M-row training sample")
    sample = train.sample(n=min(SAMPLE_SIZE, len(train)), random_state=42)
    sample.reset_index(drop=True).to_parquet(
        DATA_DIR / "sample_1M.parquet", index=False
    )
    print(f"  sample_1M.parquet: {len(sample):,} rows")

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("aborted")