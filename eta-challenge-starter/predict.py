"""Submission interface — this is what Gobblecube's grader imports.

This version uses a multi-tier lookup (Hourly/Route/Global) instead of XGBoost.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

# Update this path to match your saved lookup artifact
_MODEL_PATH = Path(__file__).parent / "hourly_lookup_model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    # This now loads the dictionary containing our 3 tiers of medians
    _MODEL_DATA = pickle.load(_f)

_HOURLY_MEDIANS = _MODEL_DATA["hourly_zone_medians"]
_ZONE_MEDIANS = _MODEL_DATA["zone_medians"]
_GLOBAL_MEDIAN = _MODEL_DATA["global_median"]


def predict(request: dict) -> float:
    """Predict trip duration using tiered median lookups."""
    
    # 1. Preprocessing (Extracting features from the 4-item request)
    pz = int(request["pickup_zone"])
    dz = int(request["dropoff_zone"])
    ts = datetime.fromisoformat(request["requested_at"])
    hour = ts.hour

    # 2. Tier 1: Exact Match (Route + Hour)
    # We use .get() because dictionaries return None if the key doesn't exist
    prediction = _HOURLY_MEDIANS.get((pz, dz, hour))
    if prediction is not None:
        return float(prediction)

    # 3. Tier 2: Route Fallback (Ignore the hour)
    prediction = _ZONE_MEDIANS.get((pz, dz))
    if prediction is not None:
        return float(prediction)

    # 4. Tier 3: Global Safety Net
    return float(_GLOBAL_MEDIAN)