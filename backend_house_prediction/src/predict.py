"""Load the trained bundle and turn requests into price predictions.

The model is cached after first load so the FastAPI process reads the pickle once.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.preprocessing import features_from_request, outward_code

DEFAULT_MODEL_PATH = config.MODEL_PATH

# Fallback residual spread, used only if an older bundle predates the field.
# Because the model is fitted in log space, +/- sigma becomes a multiplicative
# band in GBP: asymmetric, which is how price uncertainty actually behaves.
FALLBACK_LOG_RESIDUAL_STD = 0.3435


@lru_cache(maxsize=1)
def load_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load and cache the joblib bundle written by `train.py`."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_frame(features: pd.DataFrame, model_path: str | Path = DEFAULT_MODEL_PATH) -> np.ndarray:
    """Predict prices in GBP for an already-built feature frame."""
    bundle = load_bundle(model_path)
    return np.expm1(bundle["model"].predict(features[bundle["feature_columns"]]))


def resolve_location(postcode: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict | None:
    """Look up town / district / county for a postcode, or None if unknown.

    Values come from the training data itself, so anything returned here is
    guaranteed to be in the target encoder's vocabulary.
    """
    lookup = load_bundle(model_path).get("location_lookup") or {}
    entry = lookup.get(outward_code(postcode))
    return {"outward_code": outward_code(postcode), **entry} if entry else None


def predict_one(
    postcode: str,
    property_type: str,
    old_new: str,
    duration: str,
    year: int,
    month: int,
    town_city: str | None = None,
    district: str | None = None,
    county: str | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Predict a single property price, with a rough confidence interval.

    `town_city` / `district` / `county` are optional: when omitted they are resolved
    from the postcode against the training data, which is both easier for the caller
    and safer than trusting free text that may not match the encoder's vocabulary.
    """
    bundle = load_bundle(model_path)
    resolved = resolve_location(postcode, model_path)

    # Explicit values win; otherwise fall back to the lookup, then to a neutral
    # placeholder that the target encoder maps to its global mean.
    location_source = "provided" if all([town_city, district, county]) else (
        "postcode_lookup" if resolved else "unknown_postcode"
    )
    town_city = town_city or (resolved or {}).get("town_city", "UNKNOWN")
    district = district or (resolved or {}).get("district", "UNKNOWN")
    county = county or (resolved or {}).get("county", "UNKNOWN")

    features = features_from_request(
        postcode=postcode, property_type=property_type, old_new=old_new,
        duration=duration, town_city=town_city, district=district,
        county=county, year=year, month=month,
    )
    price = float(predict_frame(features, model_path)[0])

    sigma = float(bundle.get("log_residual_std") or FALLBACK_LOG_RESIDUAL_STD)
    log_price = np.log1p(price)
    return {
        "predicted_price": round(price, 2),
        "lower_bound": round(float(np.expm1(log_price - sigma)), 2),
        "upper_bound": round(float(np.expm1(log_price + sigma)), 2),
        "model_name": bundle["model_name"],
        "location_source": location_source,
        "town_city": town_city,
        "district": district,
        "county": county,
    }


if __name__ == "__main__":
    example = predict_one(
        postcode="NW10 0DY", property_type="F", old_new="N", duration="L",
        town_city="LONDON", district="BRENT", county="GREATER LONDON",
        year=2026, month=6,
    )
    for key, value in example.items():
        print(f"{key:18s}: {value}")
