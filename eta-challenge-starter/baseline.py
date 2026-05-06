#!/usr/bin/env python
"""Hybrid Baseline + OSRM: Median Lookup + LightGBM on residuals (init_score).

Trains in a few minutes on a laptop CPU. Produces `model.pkl` which `predict.py`
loads at inference. This approach calculates a 5-tier fallback median, merges
static OSRM routing data, clusters zones via K-Means, and trains a LightGBM model 
using the baselines as an `init_score` (Base Margin) to predict the residual.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans

# --- Configuration ---
DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

# Expanded Categorical Features
CATEGORICAL_FEATURES = [
    "pickup_zone", "dropoff_zone", "hour", "dayofweek", 
    "month", "is_weekend", "week_of_year", "week_of_month",
    "pickup_cluster", "dropoff_cluster",
    "cluster_route", "time_of_day",  
]

# Expanded Numeric Features 
NUMERIC_FEATURES = [
    "osrm_time", 
    "osrm_distance", 
    "base_pred",              
    "baseline_vs_osrm"       
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main() -> None:
    # --- 0. Data Loading & Clustering ---
    train_path = DATA_DIR / "sample_1M.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    osrm_path = DATA_DIR / "osrm_matrix.csv" 
    coords_path = DATA_DIR / "zone_coords.csv"
    
    for p in (train_path, dev_path, osrm_path, coords_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Ensure train, dev, osrm_matrix, and zone_coords are in the data dir."
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

    # Load Coordinates and generate Clusters
    coords_df = pd.read_csv(coords_path)
    print(f"\nLoading zone coords: {len(coords_df):,} locations loaded")
    
    # K-Means Clustering (45 clusters)
    kmeans = KMeans(n_clusters=45, random_state=42, n_init=10)
    coords_df['cluster'] = kmeans.fit_predict(coords_df[['lat', 'lon']])
    cluster_mapping = coords_df.set_index('LocationID')['cluster'].to_dict()
    
    cluster_csv_path = DATA_DIR / "zone_clusters_45.csv"
    coords_df[['LocationID', 'lat', 'lon', 'cluster']].to_csv(cluster_csv_path, index=False)
    print(f"  Saved 45-cluster mapping to {cluster_csv_path}")

    # --- 1. Feature Engineering (Time, Clusters & Interactions) ---
    print("\nEngineering time and interaction features...")
    t0_feat = time.time()
    for df in (train, dev):
        # Time Features
        df["requested_at"] = pd.to_datetime(df["requested_at"])
        df["hour"] = df["requested_at"].dt.hour.astype("int8")
        # Existing time features
        df["hour"] = df["requested_at"].dt.hour.astype("int8")
        

        df["dayofweek"] = df["requested_at"].dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8") 
        df["month"] = df["requested_at"].dt.month.astype("int8")
        df["week_of_year"] = df["requested_at"].dt.isocalendar().week.astype("int8")
        df["week_of_month"] = ((df["requested_at"].dt.day - 1) // 7 + 1).astype("int8")

        # Time of Day Binning (Converted to Integer: 0, 1, 2, 3)
        df["time_of_day"] = (df["hour"] // 6).astype("int8")
         
        # Explicit Airport Pickup and Dropoff Flags
        df["airport_pickup"] = df["pickup_zone"].isin([132, 138]).astype("int8")
        df["airport_dropoff"] = df["dropoff_zone"].isin([132, 138]).astype("int8")

        # Cluster Features mapping
        df["pickup_cluster"] = df["pickup_zone"].map(cluster_mapping).fillna(-1).astype("int8")
        df["dropoff_cluster"] = df["dropoff_zone"].map(cluster_mapping).fillna(-1).astype("int8")
        
        # Regional Route Interaction (Converted to Unique Integer)
        # Shift by +1 to handle -1 nulls, multiply by 100 to prevent collisions
        df["cluster_route"] = (
            ((df["pickup_cluster"].astype("int32") + 1) * 100) + 
            (df["dropoff_cluster"].astype("int32") + 1)
        )
        
    print(f"  features extracted in {time.time() - t0_feat:.1f}s")

    # --- 2. Vectorized 5-Tier Baseline Generation ---
    print("\nBuilding 5-Tier baseline lookups...")
    t0_base = time.time()
    
    global_median = float(train["duration_seconds"].median())
    
    # 1. Hourly Zone Route
    hourly_med_df = train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"].median().reset_index(name="hourly_med")
    # 2. Overall Zone Route
    route_med_df = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].median().reset_index(name="route_med")
    # 3. Hourly Cluster Route
    hourly_cluster_med_df = train.groupby(["pickup_cluster", "dropoff_cluster", "hour"])["duration_seconds"].median().reset_index(name="hourly_cluster_med")
    # 4. Overall Cluster Route
    cluster_med_df = train.groupby(["pickup_cluster", "dropoff_cluster"])["duration_seconds"].median().reset_index(name="cluster_med")
    
    lookup_dicts = {
        "global": global_median,
        "hourly_zone": hourly_med_df.set_index(["pickup_zone", "dropoff_zone", "hour"])["hourly_med"].to_dict(),
        "zone_route": route_med_df.set_index(["pickup_zone", "dropoff_zone"])["route_med"].to_dict(),
        "hourly_cluster": hourly_cluster_med_df.set_index(["pickup_cluster", "dropoff_cluster", "hour"])["hourly_cluster_med"].to_dict(),
        "cluster_route": cluster_med_df.set_index(["pickup_cluster", "dropoff_cluster"])["cluster_med"].to_dict()
    }

    # Apply Fallbacks & Calculate Interaction Features
    for df_name, df in [("train", train), ("dev", dev)]:
        # Merge all tiers
        df = df.merge(hourly_med_df, on=["pickup_zone", "dropoff_zone", "hour"], how="left")
        df = df.merge(route_med_df, on=["pickup_zone", "dropoff_zone"], how="left")
        df = df.merge(hourly_cluster_med_df, on=["pickup_cluster", "dropoff_cluster", "hour"], how="left")
        df = df.merge(cluster_med_df, on=["pickup_cluster", "dropoff_cluster"], how="left")
        df = df.merge(osrm_df, on=["pickup_zone", "dropoff_zone"], how="left")
        
        # Apply 5-Tier Fallback Cascade
        df["base_pred"] = (
            df["hourly_med"]
            .fillna(df["route_med"])
            .fillna(df["hourly_cluster_med"])
            .fillna(df["cluster_med"])
            .fillna(df["osrm_time"])
            .fillna(global_median)
        )
        
        # REMOVED: df["target_residual"] = df["duration_seconds"] - df["base_pred"]
        
        # Create the Interaction Features 
        df["baseline_vs_osrm"] = df["base_pred"] - df["osrm_time"]
        
        # Clean up temporary columns
        df.drop(columns=["hourly_med", "route_med", "hourly_cluster_med", "cluster_med"], inplace=True)
        
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

    # --- 4. Train LightGBM with init_score ---
    print("\nTraining LightGBM using baselines as init_score...")
    
    # CHANGED: Added init_score and changed label to actual duration
    train_data = lgb.Dataset(
        train[FEATURES], 
        label=train["duration_seconds"], 
        init_score=train["base_pred"],     # <-- The magic happens here
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False
    )
    
    dev_data = lgb.Dataset(
        dev[FEATURES], 
        label=dev["duration_seconds"], 
        init_score=dev["base_pred"],       # <-- And here
        reference=train_data,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False
    )

    params = {
        "objective": "mse",  # CHANGED: Must be MSE to escape the Median-Zero trap
        "metric": "mae",     # Keeping MAE so your evaluation matches
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
        print(f"  {row['Feature']:<25}: {row['Importance']:.2f}")

    # --- 6. Evaluate combined model on Dev ---
    print("\nEvaluating final combined model...")
    
    # CHANGED: bst.predict() outputs the raw tree sums (the residual). 
    # We must manually add the init_score (base_pred) back to get the final duration.
    lgb_residual_preds = bst.predict(dev[FEATURES])
    final_preds = dev["base_pred"] + lgb_residual_preds
    
    # Prevent negative predictions
    final_preds = np.clip(final_preds, a_min=1, a_max=None)
    
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
    
    # CHANGED: Removed target_residual
    display_cols = [
        "cluster_route", "pickup_zone", "dropoff_zone", 
         "hour", "base_pred", 
        "osrm_distance", "osrm_time", "duration_seconds", 
        "final_pred", "abs_error"
    ]
    print(worst_20[display_cols].to_string(index=False))

    # --- 8. Save Artifacts ---
    osrm_dict = osrm_df.set_index(["pickup_zone", "dropoff_zone"])[["osrm_time", "osrm_distance"]].to_dict('index')

    model_artifact = {
        "lookups": lookup_dicts,
        "osrm": osrm_dict,
        "cluster_mapping": cluster_mapping,  
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