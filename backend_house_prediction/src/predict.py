"""Load the trained bundle and turn requests into price predictions.

MODEL COMES FROM HERE, AT RUNTIME:
    train.py writes house_price_model.pkl  ->  load_bundle() reads it once
    (cached by lru_cache)                  ->  predict_one() uses it per request.

The model is cached after first load so the FastAPI process reads the pickle once.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.preprocessing import features_from_request, outward_code

DEFAULT_MODEL_PATH = config.MODEL_PATH

# Fallback residual spread, used only if an older bundle predates the field.
# Because the model is fitted in log space, +/- sigma becomes a multiplicative
# band in GBP: asymmetric, which is how price uncertainty actually behaves.
FALLBACK_LOG_RESIDUAL_STD = 0.3435


def load_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load and cache the joblib bundle written by `train.py`.

    The path is resolved to a canonical string before it reaches the cache, so
    `load_bundle()` and `load_bundle(DEFAULT_MODEL_PATH)` hit the SAME entry.
    Caching on the raw argument instead would give those two call styles different
    keys and re-unpickle the model on every alternation.
    """
    return _load_bundle_cached(str(Path(model_path).resolve()))


@lru_cache(maxsize=2)
def _load_bundle_cached(resolved_path: str) -> dict:
    """A pickled sklearn Pipeline is not portable across sklearn minor versions.
    When they disagree the raw failure is an opaque `ModuleNotFoundError: No module
    named '_loss'`, so translate it into something that names the actual problem.
    """
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.train` first."
        )

    try:
        bundle = joblib.load(path)
    except (ModuleNotFoundError, AttributeError, ImportError) as exc:
        raise RuntimeError(
            f"Could not load {path.name}: it was pickled by a different "
            f"scikit-learn version than the installed one ({sklearn.__version__}). "
            f"Original error: {exc}. "
            f"Fix: retrain with `python -m src.train` using the pinned version in "
            f"requirements.txt, then redeploy the regenerated model."
        ) from exc

    built_with = bundle.get("sklearn_version")
    if built_with and built_with != sklearn.__version__:
        # It loaded, but silently-wrong predictions are worse than a hard failure.
        raise RuntimeError(
            f"Model was trained with scikit-learn {built_with} but "
            f"{sklearn.__version__} is installed. Retrain with "
            f"`python -m src.train` and redeploy, or pin scikit-learn=={built_with}."
        )
    return bundle


def predict_frame(features: pd.DataFrame, model_path: str | Path = DEFAULT_MODEL_PATH) -> np.ndarray:
    """Predict prices in GBP for an already-built feature frame.

    THE MODEL IS USED HERE — one line does the whole scoring:
        bundle['model']      = fitted sklearn Pipeline (preprocessor + regressor)
        .predict(features)   = runs StandardScaler/OneHot/TargetEncoder, then the
                               HistGradientBoostingRegressor, returning log1p(price)
        np.expm1(...)        = maps log back to raw GBP
    `features[bundle["feature_columns"]]` guarantees the columns arrive in the exact
    order the pipeline was fitted on.
    """
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

    # Assemble the SAME 11 columns the model was trained on (see add_features /
    # features_from_request in preprocessing.py), then score them in one call.
    features = features_from_request(
        postcode=postcode, property_type=property_type, old_new=old_new,
        duration=duration, town_city=town_city, district=district,
        county=county, year=year, month=month,
    )
    price = float(predict_frame(features, model_path)[0])

    # Rough confidence band: ±1 residual sigma in LOG-space. Because sigma is
    # applied before expm1, the GBP band is asymmetric (wider above) — matching
    # how real price uncertainty behaves. sigma comes from training-time residuals.
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
