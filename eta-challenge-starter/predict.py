#!/usr/bin/env python
"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. 
"""

from __future__ import annotations

import pickle
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

_LGBM_MODEL = _MODEL_DATA["lgbm_model"]


def predict(request: dict) -> float:
    """Predict trip duration using tiered median lookups + LightGBM residuals."""
    
    # --- 1. Preprocessing ---
    pz = int(request["pickup_zone"])
    dz = int(request["dropoff_zone"])
    ts = datetime.fromisoformat(request["requested_at"])
    
    hour = ts.hour
    dow = ts.weekday() 
    month = ts.month
    is_weekend = int(dow >= 5)
    passenger_count = int(request.get("passenger_count", 1))

    # --- 2. OSRM Features ---
    osrm_dist = float(request.get("osrm_distance_meters", 0.0))
    osrm_time = float(request.get("osrm_duration_seconds", 0.0))

    # --- 3. Base Prediction (3-Tier Fallback) ---
    base_pred = _HOURLY_MEDIANS.get((pz, dz, hour))
    if base_pred is None:
        base_pred = _ROUTE_MEDIANS.get((pz, dz))
        if base_pred is None:
            base_pred = _GLOBAL_MEDIAN
            
    # --- 4. NEW: Interaction Features ---
    # We must calculate the exact same features we trained on
    baseline_vs_osrm = base_pred - osrm_time
    time_ratio = base_pred / (osrm_time + 1.0)
            
    # --- 5. LightGBM Residual Prediction ---
    # Feature order MUST match the training FEATURES list EXACTLY: 
    # Categorical: pickup_zone, dropoff_zone, hour, dayofweek, month, is_weekend
    # Numeric: passenger_count, osrm_time, osrm_distance, base_pred, baseline_vs_osrm, time_ratio
    features = [[
        pz, dz, hour, dow, month, is_weekend, 
        passenger_count, osrm_time, osrm_dist, base_pred, baseline_vs_osrm, time_ratio
    ]]
    
    residual_pred = _LGBM_MODEL.predict(features)[0]

    # --- 6. Final Combination ---
    # Final Result = Median Baseline + ML Refinement
    return float(base_pred + residual_pred)