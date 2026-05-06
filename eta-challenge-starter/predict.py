#!/usr/bin/env python
"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. 
"""

from __future__ import annotations

import pickle
import math
from datetime import datetime
from pathlib import Path

_MODEL_PATH = Path(__file__).parent / "model.pkl"

# Load the model and dictionaries once into memory
with open(_MODEL_PATH, "rb") as _f:
    _MODEL_DATA = pickle.load(_f)

_LOOKUPS = _MODEL_DATA["lookups"]
_GLOBAL_MEDIAN = _LOOKUPS["global"]
_HOURLY_ZONE = _LOOKUPS["hourly_zone"]
_ZONE_ROUTE = _LOOKUPS["zone_route"]
_HOURLY_CLUSTER = _LOOKUPS["hourly_cluster"]
_CLUSTER_ROUTE = _LOOKUPS["cluster_route"]

_OSRM = _MODEL_DATA["osrm"]
_CLUSTER_MAPPING = _MODEL_DATA["cluster_mapping"]
_LGBM_MODEL = _MODEL_DATA["lgbm_model"]

def predict(request: dict) -> float:
    """Predict trip duration using tiered median lookups + LightGBM residuals."""
    
    # --- 1. Fast Preprocessing ---
    pz = int(request.get("pickup_zone") or request.get("PULocationID"))
    dz = int(request.get("dropoff_zone") or request.get("DOLocationID"))
    
    ts_str = request.get("requested_at") or request.get("tpep_pickup_datetime")
    ts = datetime.fromisoformat(str(ts_str))
    
    hour = ts.hour
    dow = ts.weekday() 
    
    # --- 2. Lookups & Cluster Math ---
    pc = _CLUSTER_MAPPING.get(pz, -1)
    dc = _CLUSTER_MAPPING.get(dz, -1)
    
    # Calculate interaction features exactly as done in baseline.py
    time_of_day = hour // 6
    cluster_route = ((pc + 1) * 100) + (dc + 1)
    
    # Explicit Airport Pickup and Dropoff Flags
    airport_pickup = 1 if pz in (132, 138) else 0
    airport_dropoff = 1 if dz in (132, 138) else 0

    osrm_route = _OSRM.get((pz, dz))
    if osrm_route:
        osrm_time = float(osrm_route["osrm_time"])
        osrm_dist = float(osrm_route["osrm_distance"])
    else:
        osrm_time = math.nan
        osrm_dist = math.nan

    # --- 3. Base Prediction (Fallback Cascade) ---
    base_pred = _HOURLY_ZONE.get((pz, dz, hour))
    if base_pred is None:
        base_pred = _ZONE_ROUTE.get((pz, dz))
        if base_pred is None:
            base_pred = _HOURLY_CLUSTER.get((pc, dc, hour))
            if base_pred is None:
                base_pred = _CLUSTER_ROUTE.get((pc, dc), _GLOBAL_MEDIAN)
            
    # --- 4. Assemble Raw Values (NO PANDAS NEEDED) ---
    # MUST MATCH the exact order of CATEGORICAL_FEATURES + NUMERIC_FEATURES from training
    row_values = [
        # Categorical Features
        pz, 
        dz, 
        hour, 
        dow, 
        ts.month, 
        1 if dow >= 5 else 0,              # is_weekend
        ts.isocalendar()[1],               # week_of_year
        (ts.day - 1) // 7 + 1,             # week_of_month
        pc, 
        dc, 
        cluster_route,                     # integer math from above
        time_of_day,                       # integer math from above
        
        # Numeric Features
        osrm_time, 
        osrm_dist, 
        base_pred, 
        base_pred - osrm_time if not math.isnan(osrm_time) else math.nan # baseline_vs_osrm
    ]
    
    # --- 5. Lightning Fast Prediction ---
    # Because the model was trained with init_score, predict() naturally 
    # outputs the tree sums (the residual).
    residual_pred = _LGBM_MODEL.predict([row_values])[0]

    # We manually add the base_pred back to the residual to get the final duration
    return max(1.0, float(base_pred + residual_pred))