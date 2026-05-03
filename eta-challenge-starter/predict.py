#!/usr/bin/env python
"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
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
# Expected feature order from training: 
# ["pickup_zone", "dropoff_zone", "hour", "dayofweek", "month", "passenger_count"]

def predict(request: dict) -> float:
    """Predict trip duration using tiered median lookups + LightGBM residuals."""
    
    # --- 1. Preprocessing ---
    pz = int(request["pickup_zone"])
    dz = int(request["dropoff_zone"])
    ts = datetime.fromisoformat(request["requested_at"])
    
    hour = ts.hour
    dow = ts.weekday()  # .weekday() maps to 0-6, matching pandas .dt.dayofweek
    month = ts.month
    passenger_count = int(request.get("passenger_count", 1))

    # --- 2. Base Prediction (3-Tier Fallback) ---
    base_pred = _HOURLY_MEDIANS.get((pz, dz, hour))
    if base_pred is None:
        base_pred = _ROUTE_MEDIANS.get((pz, dz))
        if base_pred is None:
            base_pred = _GLOBAL_MEDIAN
            
    # --- 3. LightGBM Residual Prediction ---
    # We pass a 2D list to avoid pandas dataframe overhead at inference time.
    # LightGBM handles the raw integers perfectly because it relies on column index order.
    features = [[pz, dz, hour, dow, month, passenger_count]]
    residual_pred = _LGBM_MODEL.predict(features)[0]

    # --- 4. Final Combination ---
    return float(base_pred + residual_pred)