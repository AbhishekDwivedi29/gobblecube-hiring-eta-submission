#!/usr/bin/env python
"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. 
"""

from __future__ import annotations

import pickle
import math
from datetime import datetime
from pathlib import Path

# Update this path to match your saved lookup artifact
_MODEL_PATH = Path(__file__).parent / "model.pkl"

# Load the model and dictionaries once into memory when the module is imported
with open(_MODEL_PATH, "rb") as _f:
    _MODEL_DATA = pickle.load(_f)

_LOOKUPS = _MODEL_DATA["lookups"]
_HOURLY_MEDIANS = _LOOKUPS["hourly"]
_ROUTE_MEDIANS = _LOOKUPS["route"]
_GLOBAL_MEDIAN = _LOOKUPS["global"]

# NEW: Load the OSRM dictionary we saved during training
_OSRM = _MODEL_DATA["osrm"]

_LGBM_MODEL = _MODEL_DATA["lgbm_model"]


def predict(request: dict) -> float:
    """Predict trip duration using tiered median lookups + LightGBM residuals."""
    
    # --- 1. Preprocessing ---
    # Handle both raw parquet names (PULocationID) or clean names (pickup_zone)
    pz = int(request.get("pickup_zone", request.get("PULocationID")))
    dz = int(request.get("dropoff_zone", request.get("DOLocationID")))
    
    ts_str = request.get("requested_at", request.get("tpep_pickup_datetime"))
    ts = datetime.fromisoformat(str(ts_str))
    
    # Time features
    hour = ts.hour
    dow = ts.weekday() 
    month = ts.month
    is_weekend = int(dow >= 5)
    
    # Calculate week of year and week of month to match training data
    week_of_year = int(ts.isocalendar()[1])
    week_of_month = int((ts.day - 1) // 7 + 1)

    # --- 2. OSRM Features (LOOKUP FROM DICTIONARY) ---
    osrm_route = _OSRM.get((pz, dz))
    if osrm_route is not None:
        osrm_time = float(osrm_route["osrm_time"])
        osrm_dist = float(osrm_route["osrm_distance"])
    else:
        # Crucial: Use NaN, NOT 0.0, so LightGBM knows the data is missing
        osrm_time = math.nan
        osrm_dist = math.nan

    # --- 3. Base Prediction (3-Tier Fallback) ---
    base_pred = _HOURLY_MEDIANS.get((pz, dz, hour))
    if base_pred is None:
        base_pred = _ROUTE_MEDIANS.get((pz, dz))
        if base_pred is None:
            base_pred = _GLOBAL_MEDIAN
            
    # --- 4. Interaction Features ---
    baseline_vs_osrm = base_pred - osrm_time
            
    # --- 5. LightGBM Residual Prediction ---
    # Feature order MUST match the training FEATURES list EXACTLY: 
    # Categorical: pickup_zone, dropoff_zone, hour, dayofweek, month, is_weekend, week_of_year, week_of_month
    # Numeric: osrm_time, osrm_distance, base_pred, baseline_vs_osrm
    features = [[
        pz, dz, hour, dow, month, is_weekend, week_of_year, week_of_month,
        osrm_time, osrm_dist, base_pred, baseline_vs_osrm
    ]]
    
    residual_pred = _LGBM_MODEL.predict(features)[0]

    # --- 6. Final Combination ---
    # Final Result = Median Baseline + ML Refinement
    return float(base_pred + residual_pred)