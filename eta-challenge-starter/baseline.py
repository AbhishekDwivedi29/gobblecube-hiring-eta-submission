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

# Categorical Features (Restored weather flags)
CATEGORICAL_FEATURES = [
    "pickup_zone", "dropoff_zone", "hour", "dayofweek", 
    "month", "is_weekend", "week_of_year", "week_of_month",
    "pickup_cluster", "dropoff_cluster",
    "cluster_route", "time_of_day",
    "is_raining", "is_snowing"         
]

# Numeric Features (Updated to day-specific frequency)
NUMERIC_FEATURES = [
    "osrm_time", 
    "osrm_distance", 
    "base_pred",              
    "baseline_vs_osrm",       
    "temp_c",                  
    "windspeed_mps",           
    "visibility_m",
    "route_day_hour_freq"      # NEW: Day-of-week specific hourly route frequency
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main() -> None:
    # --- 0. Data Loading & Clustering ---
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    osrm_path = DATA_DIR / "osrm_matrix.csv" 
    coords_path = DATA_DIR / "zone_coords.csv"
    weather_path = DATA_DIR / "nyc_weather.csv"  
    
    for p in (train_path, dev_path, osrm_path, coords_path, weather_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Ensure all data files are in the data dir."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    
    # Load and clean up OSRM naming
    osrm_df = pd.read_csv(osrm_path)
    osrm_df.rename(columns={
        'osrm_duration_seconds': 'osrm_time', 
        'osrm_distance_meters': 'osrm_distance'
    }, inplace=True)
    
    # Load and clean Weather Data
    print("Preparing weather data...")
    weather_df = pd.read_csv(weather_path)
    weather_df["merge_hour"] = pd.to_datetime(weather_df["Hour"]).dt.floor('H')
    weather_df.rename(columns={
        'Temp_C': 'temp_c',
        'WindSpeed_mps': 'windspeed_mps',
        'Visibility_m': 'visibility_m',
        'Is_Raining': 'is_raining',        
        'Is_Snowing': 'is_snowing'         
    }, inplace=True)
    
    # Keep only what we need for the merge
    weather_df = weather_df[['merge_hour', 'LocationID', 'temp_c', 'windspeed_mps', 'visibility_m', 'is_raining', 'is_snowing']]
    
    # Cast booleans to integers for modeling
    weather_df['is_raining'] = weather_df['is_raining'].astype("int8")
    weather_df['is_snowing'] = weather_df['is_snowing'].astype("int8")

    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")
    print(f"  osrm:  {len(osrm_df):,} zone pairs")
    print(f"  weather: {len(weather_df):,} hourly records")

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
    for df_name, df in [("train", train), ("dev", dev)]:
        # Time Features
        df["requested_at"] = pd.to_datetime(df["requested_at"])
        df["hour"] = df["requested_at"].dt.hour.astype("int8")
        df["dayofweek"] = df["requested_at"].dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8") 
        df["month"] = df["requested_at"].dt.month.astype("int8")
        df["week_of_year"] = df["requested_at"].dt.isocalendar().week.astype("int8")
        df["week_of_month"] = ((df["requested_at"].dt.day - 1) // 7 + 1).astype("int8")
        
        # Exact hour for weather merge
        df["merge_hour"] = df["requested_at"].dt.floor('H')

        # Time of Day Binning
        df["time_of_day"] = (df["hour"] // 6).astype("int8")
         

        # Cluster Features mapping
        df["pickup_cluster"] = df["pickup_zone"].map(cluster_mapping).fillna(-1).astype("int8")
        df["dropoff_cluster"] = df["dropoff_zone"].map(cluster_mapping).fillna(-1).astype("int8")
        
        # Regional Route Interaction
        df["cluster_route"] = (
            ((df["pickup_cluster"].astype("int32") + 1) * 100) + 
            (df["dropoff_cluster"].astype("int32") + 1)
        )

        # Merge Weather Data on Time and Pickup Zone
        df = df.merge(weather_df, left_on=["merge_hour", "pickup_zone"], right_on=["merge_hour", "LocationID"], how="left")
        
        # Fill missing weather
        df['temp_c'] = df['temp_c'].fillna(weather_df['temp_c'].median()).astype("float32")
        df['windspeed_mps'] = df['windspeed_mps'].fillna(weather_df['windspeed_mps'].median()).astype("float32")
        df['visibility_m'] = df['visibility_m'].fillna(weather_df['visibility_m'].median()).astype("float32")
        df['is_raining'] = df['is_raining'].fillna(0).astype("int8")
        df['is_snowing'] = df['is_snowing'].fillna(0).astype("int8")
        
        # Clean up temporary joins
        df.drop(columns=["merge_hour", "LocationID"], inplace=True)
        
        if df_name == "train": train = df
        else: dev = df
        
    print(f"  features extracted in {time.time() - t0_feat:.1f}s")

    # --- 2. Vectorized 5-Tier Baseline Generation & Frequency ---
    print("\nBuilding lookups and frequencies...")
    t0_base = time.time()
    
    global_median = float(train["duration_seconds"].median())
    
    # Baseline Medians
    hourly_med_df = train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"].median().reset_index(name="hourly_med")
    route_med_df = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].median().reset_index(name="route_med")
    hourly_cluster_med_df = train.groupby(["pickup_cluster", "dropoff_cluster", "hour"])["duration_seconds"].median().reset_index(name="hourly_cluster_med")
    cluster_med_df = train.groupby(["pickup_cluster", "dropoff_cluster"])["duration_seconds"].median().reset_index(name="cluster_med")
    
    # NEW: Historical frequency of rides per route per hour, split by DAY OF THE WEEK
    route_freq_df = train.groupby(["pickup_zone", "dropoff_zone", "dayofweek", "hour"]).size().reset_index(name="route_day_hour_freq")

    lookup_dicts = {
        "global": global_median,
        "hourly_zone": hourly_med_df.set_index(["pickup_zone", "dropoff_zone", "hour"])["hourly_med"].to_dict(),
        "zone_route": route_med_df.set_index(["pickup_zone", "dropoff_zone"])["route_med"].to_dict(),
        "hourly_cluster": hourly_cluster_med_df.set_index(["pickup_cluster", "dropoff_cluster", "hour"])["hourly_cluster_med"].to_dict(),
        "cluster_route": cluster_med_df.set_index(["pickup_cluster", "dropoff_cluster"])["cluster_med"].to_dict(),
        # Updated to include dayofweek in the index key
        "route_day_hour_freq": route_freq_df.set_index(["pickup_zone", "dropoff_zone", "dayofweek", "hour"])["route_day_hour_freq"].to_dict() 
    }

    # Apply Fallbacks & Calculate Interaction Features
    for df_name, df in [("train", train), ("dev", dev)]:
        df = df.merge(hourly_med_df, on=["pickup_zone", "dropoff_zone", "hour"], how="left")
        df = df.merge(route_med_df, on=["pickup_zone", "dropoff_zone"], how="left")
        df = df.merge(hourly_cluster_med_df, on=["pickup_cluster", "dropoff_cluster", "hour"], how="left")
        df = df.merge(cluster_med_df, on=["pickup_cluster", "dropoff_cluster"], how="left")
        df = df.merge(osrm_df, on=["pickup_zone", "dropoff_zone"], how="left")
        
        # Merge updated frequency metric
        df = df.merge(route_freq_df, on=["pickup_zone", "dropoff_zone", "dayofweek", "hour"], how="left")
        
        # Fill missing frequencies with 0 
        df["route_day_hour_freq"] = df["route_day_hour_freq"].fillna(0).astype("float32")

        # Apply 5-Tier Fallback Cascade
        df["base_pred"] = (
            df["hourly_med"]
            .fillna(df["route_med"])
            .fillna(df["hourly_cluster_med"])
            .fillna(df["cluster_med"])
            .fillna(df["osrm_time"])
            .fillna(global_median)
        )
        
        # Create the Interaction Features 
        df["baseline_vs_osrm"] = df["base_pred"] - df["osrm_time"]
        
        # Clean up temporary columns
        df.drop(columns=["hourly_med", "route_med", "hourly_cluster_med", "cluster_med"], inplace=True)
        
        if df_name == "train": train = df
        else: dev = df
            
    print(f"  baselines and features merged in {time.time() - t0_base:.1f}s")

    # --- 3. Categorical Casting ---
    print("\nCasting categorical features for LightGBM...")
    for df in (train, dev):
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype("category")

    # --- 4. Train LightGBM with init_score ---
    print("\nTraining LightGBM using baselines as init_score...")
    
    train_data = lgb.Dataset(
        train[FEATURES], 
        label=train["duration_seconds"], 
        init_score=train["base_pred"],     
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False
    )
    
    dev_data = lgb.Dataset(
        dev[FEATURES], 
        label=dev["duration_seconds"], 
        init_score=dev["base_pred"],       
        reference=train_data,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False
    )

    params = {
        "objective": "mse",  
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
        print(f"  {row['Feature']:<25}: {row['Importance']:.2f}")

    # --- 6. Evaluate combined model on Dev ---
    print("\nEvaluating final combined model...")
    
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
    
    # Updated display columns to include route_day_hour_freq
    display_cols = [
        "cluster_route", "pickup_zone", "dropoff_zone", 
        "dayofweek", "hour", "base_pred", 
        "osrm_time", "duration_seconds", 
        "final_pred", "route_day_hour_freq", "is_raining", "is_snowing", "temp_c",               
        "windspeed_mps",          
        "visibility_m"  
    ]
    print(worst_20[display_cols].to_string(index=False))

    # --- 8. Save Artifacts ---
    osrm_dict = osrm_df.set_index(["pickup_zone", "dropoff_zone"])[["osrm_time", "osrm_distance"]].to_dict('index')
    
    weather_dict = weather_df.set_index(["merge_hour", "LocationID"]).to_dict('index')

    model_artifact #!/usr/bin/env python
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

# Categorical Features (Added Intra-zone and Airport flags)
CATEGORICAL_FEATURES = [
    "pickup_zone", "dropoff_zone", "hour", "dayofweek", 
    "month", "is_weekend", "week_of_year", "week_of_month",
    "pickup_cluster", "dropoff_cluster",
    "cluster_route", "time_of_day",
    "is_raining", "is_snowing",
    "is_intra_zone", "is_airport", "airport_intra"  # NEW: Airport Loop Fixes
]

# Numeric Features (Added log frequency and expected speed)
NUMERIC_FEATURES = [
    "osrm_time", 
    "osrm_distance", 
    "base_pred",              
    "baseline_vs_osrm",       
    "temp_c",                  
    "windspeed_mps",           
    "visibility_m",
    "route_day_hour_freq",     
    "log_route_freq",          # NEW: Scaled traffic magnitude
    "expected_zone_speed"      # NEW: Speed proxy for 0-distance OSRM trips
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main() -> None:
    # --- 0. Data Loading & Clustering ---
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    osrm_path = DATA_DIR / "osrm_matrix.csv" 
    coords_path = DATA_DIR / "zone_coords.csv"
    weather_path = DATA_DIR / "nyc_weather.csv"  
    
    for p in (train_path, dev_path, osrm_path, coords_path, weather_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Ensure all data files are in the data dir."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    
    # Load and clean up OSRM naming
    osrm_df = pd.read_csv(osrm_path)
    osrm_df.rename(columns={
        'osrm_duration_seconds': 'osrm_time', 
        'osrm_distance_meters': 'osrm_distance'
    }, inplace=True)
    
    # Load and clean Weather Data
    print("Preparing weather data...")
    weather_df = pd.read_csv(weather_path)
    weather_df["merge_hour"] = pd.to_datetime(weather_df["Hour"]).dt.floor('H')
    weather_df.rename(columns={
        'Temp_C': 'temp_c',
        'WindSpeed_mps': 'windspeed_mps',
        'Visibility_m': 'visibility_m',
        'Is_Raining': 'is_raining',        
        'Is_Snowing': 'is_snowing'         
    }, inplace=True)
    
    # Keep only what we need for the merge
    weather_df = weather_df[['merge_hour', 'LocationID', 'temp_c', 'windspeed_mps', 'visibility_m', 'is_raining', 'is_snowing']]
    
    # Cast booleans to integers for modeling
    weather_df['is_raining'] = weather_df['is_raining'].astype("int8")
    weather_df['is_snowing'] = weather_df['is_snowing'].astype("int8")

    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")
    print(f"  osrm:  {len(osrm_df):,} zone pairs")
    print(f"  weather: {len(weather_df):,} hourly records")

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
    for df_name, df in [("train", train), ("dev", dev)]:
        # Time Features
        df["requested_at"] = pd.to_datetime(df["requested_at"])
        df["hour"] = df["requested_at"].dt.hour.astype("int8")
        df["dayofweek"] = df["requested_at"].dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8") 
        df["month"] = df["requested_at"].dt.month.astype("int8")
        df["week_of_year"] = df["requested_at"].dt.isocalendar().week.astype("int8")
        df["week_of_month"] = ((df["requested_at"].dt.day - 1) // 7 + 1).astype("int8")
        
        # Exact hour for weather merge
        df["merge_hour"] = df["requested_at"].dt.floor('H')

        # Time of Day Binning
        df["time_of_day"] = (df["hour"] // 6).astype("int8")
         
        # Cluster Features mapping
        df["pickup_cluster"] = df["pickup_zone"].map(cluster_mapping).fillna(-1).astype("int8")
        df["dropoff_cluster"] = df["dropoff_zone"].map(cluster_mapping).fillna(-1).astype("int8")
        
        # Regional Route Interaction
        df["cluster_route"] = (
            ((df["pickup_cluster"].astype("int32") + 1) * 100) + 
            (df["dropoff_cluster"].astype("int32") + 1)
        )

        # NEW: Intra-Zone and Airport Tracking Features
        df["is_intra_zone"] = (df["pickup_zone"] == df["dropoff_zone"]).astype("int8")
        df["is_airport"] = df["pickup_zone"].isin([132, 138]).astype("int8")
        df["airport_intra"] = (df["is_intra_zone"] & df["is_airport"]).astype("int8")

        # Merge Weather Data on Time and Pickup Zone
        df = df.merge(weather_df, left_on=["merge_hour", "pickup_zone"], right_on=["merge_hour", "LocationID"], how="left")
        
        # Fill missing weather
        df['temp_c'] = df['temp_c'].fillna(weather_df['temp_c'].median()).astype("float32")
        df['windspeed_mps'] = df['windspeed_mps'].fillna(weather_df['windspeed_mps'].median()).astype("float32")
        df['visibility_m'] = df['visibility_m'].fillna(weather_df['visibility_m'].median()).astype("float32")
        df['is_raining'] = df['is_raining'].fillna(0).astype("int8")
        df['is_snowing'] = df['is_snowing'].fillna(0).astype("int8")
        
        # Clean up temporary joins
        df.drop(columns=["merge_hour", "LocationID"], inplace=True)
        
        if df_name == "train": train = df
        else: dev = df
        
    print(f"  features extracted in {time.time() - t0_feat:.1f}s")

    # --- 2. Vectorized 5-Tier Baseline Generation & Frequency ---
    print("\nBuilding lookups and frequencies...")
    t0_base = time.time()
    
    global_median = float(train["duration_seconds"].median())
    
    # Baseline Medians
    hourly_med_df = train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"].median().reset_index(name="hourly_med")
    route_med_df = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].median().reset_index(name="route_med")
    hourly_cluster_med_df = train.groupby(["pickup_cluster", "dropoff_cluster", "hour"])["duration_seconds"].median().reset_index(name="hourly_cluster_med")
    cluster_med_df = train.groupby(["pickup_cluster", "dropoff_cluster"])["duration_seconds"].median().reset_index(name="cluster_med")
    
    # Historical frequency of rides per route per hour
    route_freq_df = train.groupby(["pickup_zone", "dropoff_zone", "dayofweek", "hour"]).size().reset_index(name="route_day_hour_freq")

    lookup_dicts = {
        "global": global_median,
        "hourly_zone": hourly_med_df.set_index(["pickup_zone", "dropoff_zone", "hour"])["hourly_med"].to_dict(),
        "zone_route": route_med_df.set_index(["pickup_zone", "dropoff_zone"])["route_med"].to_dict(),
        "hourly_cluster": hourly_cluster_med_df.set_index(["pickup_cluster", "dropoff_cluster", "hour"])["hourly_cluster_med"].to_dict(),
        "cluster_route": cluster_med_df.set_index(["pickup_cluster", "dropoff_cluster"])["cluster_med"].to_dict(),
        "route_day_hour_freq": route_freq_df.set_index(["pickup_zone", "dropoff_zone", "dayofweek", "hour"])["route_day_hour_freq"].to_dict() 
    }

    # Apply Fallbacks & Calculate Interaction Features
    for df_name, df in [("train", train), ("dev", dev)]:
        df = df.merge(hourly_med_df, on=["pickup_zone", "dropoff_zone", "hour"], how="left")
        df = df.merge(route_med_df, on=["pickup_zone", "dropoff_zone"], how="left")
        df = df.merge(hourly_cluster_med_df, on=["pickup_cluster", "dropoff_cluster", "hour"], how="left")
        df = df.merge(cluster_med_df, on=["pickup_cluster", "dropoff_cluster"], how="left")
        df = df.merge(osrm_df, on=["pickup_zone", "dropoff_zone"], how="left")
        
        # Merge updated frequency metric
        df = df.merge(route_freq_df, on=["pickup_zone", "dropoff_zone", "dayofweek", "hour"], how="left")
        
        # Fill missing frequencies with 0 and create Log variant
        df["route_day_hour_freq"] = df["route_day_hour_freq"].fillna(0).astype("float32")
        df["log_route_freq"] = np.log1p(df["route_day_hour_freq"]).astype("float32")

        # Apply 5-Tier Fallback Cascade
        df["base_pred"] = (
            df["hourly_med"]
            .fillna(df["route_med"])
            .fillna(df["hourly_cluster_med"])
            .fillna(df["cluster_med"])
            .fillna(df["osrm_time"])
            .fillna(global_median)
        )
        
        # Create the Interaction Features 
        df["baseline_vs_osrm"] = df["base_pred"] - df["osrm_time"]
        
        # Clean up temporary columns
        df.drop(columns=["hourly_med", "route_med", "hourly_cluster_med", "cluster_med"], inplace=True)
        
        if df_name == "train": train = df
        else: dev = df
            
    # --- 2.5 Historical Expected Speed ---
    # Calculate historical speed on the train set (avoiding division by zero)
    train["hist_speed"] = train["osrm_distance"] / np.maximum(train["duration_seconds"], 1)
    
    # Create speed lookup based on zone and hour
    speed_df = train.groupby(["pickup_zone", "hour"])["hist_speed"].median().reset_index(name="expected_zone_speed")
    
    # Map back to train and dev
    train = train.merge(speed_df, on=["pickup_zone", "hour"], how="left")
    dev = dev.merge(speed_df, on=["pickup_zone", "hour"], how="left")
    
    # Fill missing speeds with global median
    global_med_speed = float(train["hist_speed"].median())
    train["expected_zone_speed"] = train["expected_zone_speed"].fillna(global_med_speed).astype("float32")
    dev["expected_zone_speed"] = dev["expected_zone_speed"].fillna(global_med_speed).astype("float32")

    # Clean up calculation column
    train.drop(columns=["hist_speed"], inplace=True)

    print(f"  baselines and features merged in {time.time() - t0_base:.1f}s")

    # --- 3. Categorical Casting ---
    print("\nCasting categorical features for LightGBM...")
    for df in (train, dev):
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype("category")

    # --- 4. Train LightGBM with init_score ---
    print("\nTraining LightGBM using baselines as init_score...")
    
    train_data = lgb.Dataset(
        train[FEATURES], 
        label=train["duration_seconds"], 
        init_score=train["base_pred"],     
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False
    )
    
    dev_data = lgb.Dataset(
        dev[FEATURES], 
        label=dev["duration_seconds"], 
        init_score=dev["base_pred"],       
        reference=train_data,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False
    )

    params = {
        "objective": "mse",  
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
        print(f"  {row['Feature']:<25}: {row['Importance']:.2f}")

    # --- 6. Evaluate combined model on Dev ---
    print("\nEvaluating final combined model...")
    
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
    
    # Updated display columns to include new airport features
    display_cols = [
        "pickup_zone", "dropoff_zone", "hour", "base_pred", 
        "osrm_time", "duration_seconds", "final_pred", 
        "airport_intra", "log_route_freq", "expected_zone_speed"
    ]
    print(worst_20[display_cols].to_string(index=False))

    # --- 8. Save Artifacts ---
    osrm_dict = osrm_df.set_index(["pickup_zone", "dropoff_zone"])[["osrm_time", "osrm_distance"]].to_dict('index')
    
    weather_dict = weather_df.set_index(["merge_hour", "LocationID"]).to_dict('index')
    
    # NEW: Save the speed lookup mapping for inference
    speed_lookup_dict = speed_df.set_index(["pickup_zone", "hour"])["expected_zone_speed"].to_dict()

    model_artifact = {
        "lookups": lookup_dicts,
        "speed_lookup": speed_lookup_dict,   # Added inference dependency
        "global_med_speed": global_med_speed, # Added inference dependency
        "osrm": osrm_dict,
        "weather": weather_dict,            
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
        print("aborted")= {
        "lookups": lookup_dicts,
        "osrm": osrm_dict,
        "weather": weather_dict,            
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