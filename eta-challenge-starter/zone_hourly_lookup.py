#!/usr/bin/env python
"""
Time-Aware Baseline: Zone-Pair + Hour Lookup.

This script creates a smart lookup dictionary. It tries to predict 
based on (pickup, dropoff, hour). If that exact combo is missing, 
it falls back to (pickup, dropoff). If that's missing, it uses the global median.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "hourly_lookup_model.pkl"

def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"

    if not train_path.exists() or not dev_path.exists():
        raise SystemExit("Missing data files. Run `python data/download_data.py` first.")

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("\nFeature Engineering: Extracting the Hour...")
    # Convert string timestamps to datetime objects so we can grab the hour
    train["requested_at"] = pd.to_datetime(train["requested_at"])
    train["hour"] = train["requested_at"].dt.hour
    
    dev["requested_at"] = pd.to_datetime(dev["requested_at"])
    dev["hour"] = dev["requested_at"].dt.hour

    print("\nCalculating medians...")
    t0 = time.time()

    # Tier 3: The Global Safety Net
    global_median = float(train["duration_seconds"].median())

    # Tier 2: The Route Fallback (pickup, dropoff)
    zone_medians = (
        train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .median()
        .to_dict()
    )
    
    # Tier 1: The Exact Match (pickup, dropoff, hour)
    hourly_zone_medians = (
        train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"]
        .median()
        .to_dict()
    )

    print(f"  computed {len(hourly_zone_medians):,} unique route+hour combinations in {time.time() - t0:.2f}s")

    # Create our new multi-tier "model" artifact
    model_artifact = {
        "global_median": global_median,
        "zone_medians": zone_medians,
        "hourly_zone_medians": hourly_zone_medians
    }

    # Save the lookup dictionary
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"Saved hourly lookup model to {MODEL_PATH}")

    # Quick Local Evaluation on Dev
    print("\nEvaluating on Dev...")
    
    def predict_row(row):
        # 1. Try to find the exact route at the exact hour
        exact_match = hourly_zone_medians.get((row.pickup_zone, row.dropoff_zone, row.hour))
        if exact_match is not None:
            return exact_match
            
        # 2. Fallback: I know the route, but nobody took it at this exact hour. Use route average.
        route_match = zone_medians.get((row.pickup_zone, row.dropoff_zone))
        if route_match is not None:
            return route_match
            
        # 3. Fallback: Complete guessing
        return global_median

    preds = dev.apply(predict_row, axis=1).to_numpy()
    y_dev = dev["duration_seconds"].to_numpy()

    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"Dev MAE (Hourly Lookup Baseline): {mae:.1f} seconds")


if __name__ == "__main__":
    main()