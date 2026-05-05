#!/usr/bin/env python
"""Hybrid Baseline + OSRM: Median Lookup + LightGBM on residuals.

Trains in a few minutes on a laptop CPU. Produces `model.pkl` which `predict.py`
loads at inference. This approach calculates a 3-tier fallback median, merges
static OSRM routing data, and trains a LightGBM model to predict the residual error.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# --- Configuration ---
DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

# Categorical features need specific handling in LightGBM
CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dayofweek", "month", "is_weekend"]

#  Added base_pred, osrm facts, and our calculated interactions
NUMERIC_FEATURES = [
    "passenger_count", 
    "osrm_time", 
    "osrm_distance", 
    "base_pred",          # The baseline scale
    "baseline_vs_osrm",   # Absolute traffic delay (seconds)
    "time_ratio"          # Relative traffic multiplier (e.g. 1.5x)
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES



def main() -> None:
    # --- 0. Data Loading ---
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    osrm_path = DATA_DIR / "osrm_matrix.csv" 
    
    for p in (train_path, dev_path, osrm_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Ensure train, dev, and osrm_matrix are in the data dir."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    
    # Load and clean up OSRM naming for easier use
    osrm_df = pd.read_csv(osrm_path)
    osrm_df.rename(columns={
        'osrm_duration_seconds': 'osrm_time', 
        'osrm_distance_meters': 'osrm_distance'
    }, inplace=True)
    
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")
    print(f"  osrm:  {len(osrm_df):,} zone pairs")

    # --- 1. Feature Engineering (Time) ---
    print("\nEngineering time features...")
    t0_feat = time.time()
    for df in (train, dev):
        df["requested_at"] = pd.to_datetime(df["requested_at"])
        df["hour"] = df["requested_at"].dt.hour.astype("int8")
        df["dayofweek"] = df["requested_at"].dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8") 
        df["month"] = df["requested_at"].dt.month.astype("int8")
        df["passenger_count"] = df["passenger_count"].astype("int8")
    print(f"  features extracted in {time.time() - t0_feat:.1f}s")

    # --- 2. Vectorized Baseline Generation ---
    print("\nBuilding baseline lookups and calculating base predictions...")
    t0_base = time.time()
    
    global_median = float(train["duration_seconds"].median())
    
    # Create median dataframes for fast merging
    route_med_df = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].median().reset_index(name="route_med")
    hourly_med_df = train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"].median().reset_index(name="hourly_med")
    
    lookup_dicts = {
        "global": global_median,
        "route": route_med_df.set_index(["pickup_zone", "dropoff_zone"])["route_med"].to_dict(),
        "hourly": hourly_med_df.set_index(["pickup_zone", "dropoff_zone", "hour"])["hourly_med"].to_dict()
    }

    # Vectorized 3-Tier Fallback logic & OSRM Merging
    for df_name, df in [("train", train), ("dev", dev)]:
        # Merge hourly and route medians
        df = df.merge(hourly_med_df, on=["pickup_zone", "dropoff_zone", "hour"], how="left")
        df = df.merge(route_med_df, on=["pickup_zone", "dropoff_zone"], how="left")
        
        # Apply 3-Tier Fallback: Hourly -> Route -> Global
        df["base_pred"] = df["hourly_med"].fillna(df["route_med"]).fillna(global_median)
        df["target_residual"] = df["duration_seconds"] - df["base_pred"]
        
        # ✅ Merge OSRM Data
        df = df.merge(osrm_df, on=["pickup_zone", "dropoff_zone"], how="left")
        
        # ✅ Create the Interaction Features
        # 1. Absolute difference (Traffic delay in seconds)
        df["baseline_vs_osrm"] = df["base_pred"] - df["osrm_time"]
        
        # 2. Relative ratio (Traffic congestion multiplier). Add +1.0 to prevent DivByZero.
        df["time_ratio"] = df["base_pred"] / (df["osrm_time"] + 1.0)
        
        # Clean up temporary columns
        df.drop(columns=["hourly_med", "route_med"], inplace=True)
        
        # Re-assign back to original variables
        if df_name == "train": 
            train = df
        else: 
            dev = df
            
    print(f"  baselines and OSRM merged in {time.time() - t0_base:.1f}s")

    # --- 3. Categorical Casting ---
    print("\nCasting categorical features for LightGBM...")
    for df in (train, dev):
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype("category")

    # --- 4. Train LightGBM on Residuals ---
    print("\nTraining LightGBM on residuals...")
    train_data = lgb.Dataset(
        train[FEATURES], 
        label=train["target_residual"],
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False
    )
    
    dev_data = lgb.Dataset(
        dev[FEATURES], 
        label=dev["target_residual"], 
        reference=train_data,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False
    )

    params = {
        "objective": "mae",  
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42
    }

    t0_train = time.time()
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[dev_data],
        valid_names=["dev"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    print(f"  trained in {time.time() - t0_train:.0f}s")

    # --- 5. Print Feature Importances ---
    print("\nFeature Importances (Gain):")
    importances = bst.feature_importance(importance_type='gain')
    feature_names = bst.feature_name()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"  {row['Feature']:<16}: {row['Importance']:.2f}")

    # --- 6. Evaluate combined model on Dev ---
    print("\nEvaluating final combined model...")
    lgb_residual_preds = bst.predict(dev[FEATURES])
    
    # Final Result = Median Baseline + ML Refinement
    final_preds = dev["base_pred"] + lgb_residual_preds
    
    final_mae = float(np.mean(np.abs(final_preds - dev["duration_seconds"])))
    baseline_mae = float(np.mean(np.abs(dev["base_pred"] - dev["duration_seconds"])))
    
    print(f"  Baseline MAE:     {baseline_mae:.1f} seconds")
    print(f"  Final Hybrid MAE: {final_mae:.1f} seconds")
    print(f"  Improvement:      {baseline_mae - final_mae:.1f} seconds")


    # --- 7. Deep Dive Error Analysis ---
    print("\n🔍 Deep Dive: Error Analysis")
    dev["final_pred"] = final_preds
    dev["abs_error"] = np.abs(dev["final_pred"] - dev["duration_seconds"])
    
    print("  Top 20 Worst Predictions:")
    worst_20 = dev.nlargest(20, "abs_error")
    
    # Updated columns to include dayofweek, osrm_distance, and osrm_time
    display_cols = [
        "pickup_zone", "dropoff_zone", "dayofweek", "hour", 
        "osrm_distance", "osrm_time", "duration_seconds", 
        "final_pred", "abs_error"
    ]
    print(worst_20[display_cols].to_string(index=False))

    # --- 8. Save Artifacts ---
    # Convert OSRM df to a dictionary so predict.py can map new pairs easily
    osrm_dict = osrm_df.set_index(["pickup_zone", "dropoff_zone"])[["osrm_time", "osrm_distance"]].to_dict('index')

    model_artifact = {
        "lookups": lookup_dicts,
        "osrm": osrm_dict,
        "lgbm_model": bst,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"\nSaved model artifact to {MODEL_PATH}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("aborted")