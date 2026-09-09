"""Train, evaluate and persist the house price model.

WHERE THE MODEL COMES FROM (the producer side):
    1. load_raw()  -> clean()  -> add_features()   build X, y from the CSV
    2. pick and fit the best of {Ridge, RandomForest, HistGradientBoosting}
    3. cross-validate to prove the score isn't luck
    4. REFIT THE WINNER ON 100% of the data (the hold-out was only for measuring)
    5. joblib.dump(bundle)  ->  models/house_price_model.pkl
       the bundle is the SINGLE ARTEFACT readers (predict.py / app.py) consume.

Usage:
    python -m src.train                      # train on the default dataset
    python -m src.train --no-cv              # skip 5-fold CV (faster)
    python -m src.train --data path/to.csv --out models/house_price_model.pkl
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET,
    add_features,
    build_location_lookup,
    build_preprocessor,
    clean,
    load_raw,
)

DEFAULT_DATA = config.DATA_PATH
DEFAULT_OUT = config.MODEL_PATH
RANDOM_STATE = config.RANDOM_STATE

# Prices are strongly right-skewed, so every model is trained on log1p(price)
# and predictions are mapped back with expm1.
def candidate_models() -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=RANDOM_STATE,
        ),
    }


def evaluate(model, X_train, y_train_log, X_test, y_test_log, name: str) -> dict:
    """Score one fitted model on both the log scale and the original GBP scale."""
    pred_log = model.predict(X_test)
    pred, actual = np.expm1(pred_log), np.expm1(y_test_log)
    ape = np.abs(pred - actual) / actual

    train_r2 = r2_score(y_train_log, model.predict(X_train))
    test_r2 = r2_score(y_test_log, pred_log)

    return {
        "model": name,
        "R2_log": test_r2,
        "RMSE_log": float(np.sqrt(mean_squared_error(y_test_log, pred_log))),
        "MAE_log": float(mean_absolute_error(y_test_log, pred_log)),
        "R2_GBP": float(r2_score(actual, pred)),
        "MAE_GBP": float(mean_absolute_error(actual, pred)),
        "MdAPE_pct": float(np.median(ape) * 100),
        "within_10pct": float((ape <= 0.10).mean() * 100),
        "within_20pct": float((ape <= 0.20).mean() * 100),
        # Spread of log-scale residuals; the API turns this into a prediction interval.
        "residual_std_log": float(np.std(y_test_log - pred_log)),
        "train_R2_log": float(train_r2),
        "overfit_gap": float(train_r2 - test_r2),
    }


def cross_validate_model(estimator, X, y_log, folds: int = 5) -> dict:
    """K-fold CV to confirm the hold-out score is not a lucky split."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        Pipeline([("preprocessor", build_preprocessor()), ("regressor", estimator)]),
        X, y_log, cv=cv,
        scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    )
    return {
        "folds": folds,
        "R2_per_fold": [round(float(s), 4) for s in scores["test_r2"]],
        "R2_mean": float(scores["test_r2"].mean()),
        "R2_std": float(scores["test_r2"].std()),
        "RMSE_log_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
        "MAE_log_mean": float(-scores["test_neg_mean_absolute_error"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the house price model.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}\nSee README for how to fetch it.")

    print(f"Loading {args.data} ...")
    prepared = add_features(clean(load_raw(args.data)))
    X, y = prepared[FEATURE_COLUMNS], prepared[TARGET]
    print(f"Modelling rows: {len(X):,}  |  features: {len(FEATURE_COLUMNS)}")

    # Postcode -> location map, shipped inside the bundle so the API can resolve
    # town/district/county itself instead of trusting free text from the client.
    location_lookup = build_location_lookup(prepared)
    print(f"Location lookup: {len(location_lookup):,} postcode areas")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )
    y_train_log, y_test_log = np.log1p(y_train), np.log1p(y_test)
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    results, fitted = [], {}
    for name, estimator in candidate_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("regressor", estimator)])
        started = time.time()
        pipe.fit(X_train, y_train_log)
        elapsed = time.time() - started

        row = evaluate(pipe, X_train, y_train_log, X_test, y_test_log, name)
        row["fit_seconds"] = round(elapsed, 1)
        results.append(row)
        fitted[name] = pipe
        print(f"{name:22s} R2log={row['R2_log']:.4f}  MAElog={row['MAE_log']:.4f}  "
              f"MdAPE={row['MdAPE_pct']:.1f}%  gap={row['overfit_gap']:.3f}  ({elapsed:.0f}s)")

    table = pd.DataFrame(results).set_index("model").sort_values("R2_log", ascending=False)
    best_name = str(table.index[0])
    best_model = fitted[best_name]

    print("\n" + "=" * 70)
    print(table.round(4).to_string())
    print("=" * 70)
    print(f"Best model: {best_name}  (test R2 log = {table.loc[best_name, 'R2_log']:.4f})")

    cv_report = None
    if not args.no_cv:
        print(f"\nRunning 5-fold cross-validation on {best_name} ...")
        cv_report = cross_validate_model(candidate_models()[best_name], X, np.log1p(y))
        print(f"  folds : {cv_report['R2_per_fold']}")
        print(f"  R2    = {cv_report['R2_mean']:.4f} +/- {cv_report['R2_std']:.4f}")
        drift = abs(table.loc[best_name, "R2_log"] - cv_report["R2_mean"])
        verdict = "STABLE" if drift < 2 * cv_report["R2_std"] + 0.01 else "UNSTABLE - investigate"
        print(f"  hold-out vs CV mean differs by {drift:.4f} -> {verdict}")

    # Refit the winner on 100% of the data before shipping it: the hold-out split
    # existed to measure quality, and the served model should use every row.
    print(f"\nRefitting {best_name} on the full dataset ...")
    final_model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", candidate_models()[best_name]),
    ])
    final_model.fit(X, np.log1p(y))

    # The bundle is a plain dict that travels with the model everywhere it goes.
    # predict.py unpacks it at runtime: "model" is what actually predicts,
    # feature_columns tells it how to order inputs, location_lookup lets the API
    # resolve places, and the metrics/interval fields drive the response UI.
    bundle = {
        "model": final_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target": "log1p(price)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(X)),
        "metrics": table.loc[best_name].to_dict(),
        "cross_validation": cv_report,
        "sklearn_pipeline": "StandardScaler + OneHotEncoder + TargetEncoder -> " + best_name,
        "location_lookup": location_lookup,
        "log_residual_std": float(table.loc[best_name, "residual_std_log"]),
        # Recorded so a version mismatch can be reported clearly at load time
        # instead of surfacing as a cryptic "No module named '_loss'".
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    print(f"Saved model bundle -> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    metrics_path = args.out.parent / "metrics.json"
    skip = {"model", "location_lookup"}   # the pipeline and the 2k-entry map are not metrics
    metrics_path.write_text(json.dumps(
        {k: v for k, v in bundle.items() if k not in skip}, indent=2, default=str
    ))
    print(f"Saved metrics       -> {metrics_path}")


if __name__ == "__main__":
    main()
