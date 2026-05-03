#!/usr/bin/env python
"""Hybrid Baseline: Median Lookup + LightGBM on residuals.

Trains in a few minutes on a laptop CPU. Produces `model.pkl` which `predict.py`
loads at inference. This approach calculates a 3-tier fallback median and trains 
a LightGBM model to predict the residual error.

Prerequisites:
    python data/download_data.py   # one-time, ~500 MB download

Run:
    python train_hybrid.py         # trains and saves model.pkl
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
CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dayofweek", "month"]
FEATURES = CATEGORICAL_FEATURES + ["passenger_count"]


def main() -> None:
    # --- 0. Data Loading ---
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    # --- 1. Feature Engineering ---
    print("\nEngineering features...")
    t0_feat = time.time()
    for df in (train, dev):
        df["requested_at"] = pd.to_datetime(df["requested_at"])
        df["hour"] = df["requested_at"].dt.hour.astype("int8")
        df["dayofweek"] = df["requested_at"].dt.dayofweek.astype("int8")
        df["month"] = df["requested_at"].dt.month.astype("int8")
        df["passenger_count"] = df["passenger_count"].astype("int8")
    print(f"  features extracted in {time.time() - t0_feat:.1f}s")

    # --- 2. Vectorized Baseline Generation ---
    # We do this BEFORE casting to category to ensure clean merges/groupbys
    print("\nBuilding baseline lookups and calculating base predictions...")
    t0_base = time.time()
    
    global_median = float(train["duration_seconds"].median())
    
    # Create median dataframes for fast merging
    route_med_df = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].median().reset_index(name="route_med")
    hourly_med_df = train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"].median().reset_index(name="hourly_med")
    
    # Store as dictionaries for predict.py inference (where you process 1 row at a time)
    lookup_dicts = {
        "global": global_median,
        "route": route_med_df.set_index(["pickup_zone", "dropoff_zone"])["route_med"].to_dict(),
        "hourly": hourly_med_df.set_index(["pickup_zone", "dropoff_zone", "hour"])["hourly_med"].to_dict()
    }

    # Vectorized 3-Tier Fallback logic for train and dev
    for df_name, df in [("train", train), ("dev", dev)]:
        # Merge hourly and route medians
        df = df.merge(hourly_med_df, on=["pickup_zone", "dropoff_zone", "hour"], how="left")
        df = df.merge(route_med_df, on=["pickup_zone", "dropoff_zone"], how="left")
        
        # Apply 3-Tier Fallback: Hourly -> Route -> Global
        df["base_pred"] = df["hourly_med"].fillna(df["route_med"]).fillna(global_median)
        df["target_residual"] = df["duration_seconds"] - df["base_pred"]
        
        # Clean up temporary columns
        df.drop(columns=["hourly_med", "route_med"], inplace=True)
        
        # Re-assign back to original variables
        if df_name == "train": 
            train = df
        else: 
            dev = df
            
    print(f"  baselines calculated in {time.time() - t0_base:.1f}s")

    # --- 3. Categorical Casting ---
    print("\nCasting categorical features for LightGBM...")
    for df in (train, dev):
        for col in CATEGORICAL_FEATURES:
            # Cast to 'category' dtype so LightGBM handles them natively
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


    # --- 7 . Save Artifacts ---
    model_artifact = {
        "lookups": lookup_dicts,
        "lgbm_model": bst,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"\nSaved model artifact to {MODEL_PATH}")

    

if __name__ == "__main__":
    main()
    