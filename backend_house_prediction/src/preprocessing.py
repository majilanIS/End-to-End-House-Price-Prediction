"""Data loading, cleaning and feature engineering.

Single source of truth shared by the notebook, `train.py` and the FastAPI app,
so the features built at serving time are identical to the ones used in training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# ---------------------------------------------------------------------------
# Schema of the HM Land Registry Price Paid Data file (it ships without a header)
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "transaction_id", "price", "date_of_transfer", "postcode",
    "property_type", "old_new", "duration", "paon", "saon",
    "street", "locality", "town_city", "district", "county",
    "ppd_category", "record_status",
]

# ---- Feature groups -------------------------------------------------------
# Low cardinality -> one-hot encoded.
LOW_CARD_FEATURES = ["property_type", "old_new", "duration"]
# Numeric -> standard scaled.
NUMERIC_FEATURES = ["year", "month"]
# High cardinality location -> target encoded (one-hot would explode to ~1k columns
# and, as measured, scores ~0.18 R2 lower).
HIGH_CARD_FEATURES = ["town_city", "district", "county", "area", "area_code", "sector"]

FEATURE_COLUMNS = LOW_CARD_FEATURES + NUMERIC_FEATURES + HIGH_CARD_FEATURES
TARGET = "price"

# ---- Filtering thresholds -------------------------------------------------
# Price range kept for training. Removes only ~99 rows but lifts GBP-scale R2
# from 0.25 to 0.58, because a handful of multi-million bulk transfers otherwise
# dominate the squared error.
MIN_PRICE = 20_000
MAX_PRICE = 5_000_000

# Human-readable codes, reused by the API for validation and docs.
PROPERTY_TYPES = {
    "D": "Detached",
    "S": "Semi-detached",
    "T": "Terraced",
    "F": "Flat / Maisonette",
    "O": "Other",
}
OLD_NEW = {"Y": "Newly built", "N": "Established building"}
DURATION = {"F": "Freehold", "L": "Leasehold"}


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the headerless Price Paid CSV into a DataFrame."""
    return pd.read_csv(csv_path, names=RAW_COLUMNS, quotechar='"', header=None)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot support a market-price model.

    Three filters, in order:
      1. missing postcode  -> no location signal at all
      2. record_status 'D' -> deletion records from the monthly update file
      3. ppd_category 'B'  -> not full-market-value sales (repossessions,
         buy-to-let portfolios, transfers between related parties)
      4. price outside [MIN_PRICE, MAX_PRICE] -> bulk/commercial transfers
    """
    out = df.dropna(subset=["postcode"]).copy()
    out = out[out["record_status"] != "D"]
    out = out[out["ppd_category"] == "A"]
    out = out[(out[TARGET] >= MIN_PRICE) & (out[TARGET] <= MAX_PRICE)]
    return out.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model features from the cleaned columns.

    Splits the postcode into three nested geographic levels so the model can fall
    back to a coarser one when a fine-grained code is rare or unseen:
        "NW10 0DY" -> area "NW", area_code (outward) "NW10", sector "NW10 0"
    """
    out = df.copy()

    date = pd.to_datetime(out["date_of_transfer"], errors="coerce")
    out["year"] = date.dt.year
    out["month"] = date.dt.month

    postcode = out["postcode"].astype(str).str.strip().str.upper()
    out["area_code"] = postcode.str.split().str[0]
    out["area"] = postcode.str.extract(r"^([A-Z]+)")[0]
    out["sector"] = postcode.str.replace(r"^(\S+)\s(\d).*", r"\1 \2", regex=True)

    for col in LOW_CARD_FEATURES + ["town_city", "district", "county"]:
        out[col] = out[col].astype(str).str.strip().str.upper()

    return out


def build_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Full path from raw CSV to model-ready `(X, y)`."""
    df = add_features(clean(load_raw(csv_path)))
    return df[FEATURE_COLUMNS], df[TARGET]


def outward_code(postcode: str) -> str:
    """'nw10 0dy' -> 'NW10'. Tolerates a missing space ('NW100DY')."""
    cleaned = " ".join(str(postcode).strip().upper().split())
    if " " in cleaned:
        return cleaned.split()[0]
    # No space: the inward code is always exactly 3 characters, so the rest is outward.
    return cleaned[:-3] if len(cleaned) > 3 else cleaned


def build_location_lookup(df: pd.DataFrame) -> dict[str, dict]:
    """Map each postcode outward code to its most common town / district / county.

    Why this exists: `town_city`, `district` and `county` are target-encoded, so they
    must match the training vocabulary *exactly*. A user typing "Gtr London" instead of
    "GREATER LONDON" would silently fall back to the encoder's global mean and get a
    confidently wrong answer. Deriving them from the postcode removes that failure mode.

    Expects a frame that has already been through `add_features`.
    """
    lookup: dict[str, dict] = {}
    grouped = df.groupby("area_code", observed=True)

    for code, block in grouped:
        lookup[str(code)] = {
            "town_city": block["town_city"].mode().iloc[0],
            "district": block["district"].mode().iloc[0],
            "county": block["county"].mode().iloc[0],
            # How many sales back this mapping, and how dominant the winning district is.
            # The API surfaces these so a client can flag a thin or ambiguous match.
            "n_sales": int(len(block)),
            "confidence": round(float(block["district"].value_counts(normalize=True).iloc[0]), 3),
        }
    return lookup


def build_preprocessor() -> ColumnTransformer:
    """Feature transformer. Fitted on train only, so it never leaks the test set.

    `TargetEncoder` cross-fits internally, so the encoding a row receives during
    `fit_transform` is derived from other folds rather than from its own target.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("low", OneHotEncoder(handle_unknown="ignore"), LOW_CARD_FEATURES),
            ("high", TargetEncoder(target_type="continuous", random_state=42),
             HIGH_CARD_FEATURES),
        ]
    )


def features_from_request(
    postcode: str,
    property_type: str,
    old_new: str,
    duration: str,
    town_city: str,
    district: str,
    county: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Build a one-row feature frame for a single prediction request.

    Runs the same derivations as `add_features` so serving matches training.
    """
    row = pd.DataFrame([{
        "price": 0,  # placeholder, unused
        "date_of_transfer": f"{year:04d}-{month:02d}-01",
        "postcode": postcode,
        "property_type": property_type,
        "old_new": old_new,
        "duration": duration,
        "town_city": town_city,
        "district": district,
        "county": county,
    }])
    return add_features(row)[FEATURE_COLUMNS]
