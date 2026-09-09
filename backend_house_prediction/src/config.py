"""Runtime configuration, read from environment variables / a `.env` file.

ROLE IN THE FLOW: this single module feeds BOTH sides of the pipeline —
    src/train.py           reads DATA_PATH / MODEL_PATH / RANDOM_STATE / TEST_SIZE / CV_FOLDS
    app/main.py (+predict) reads MODEL_PATH / API_HOST / API_PORT / RELOAD / ALLOWED_ORIGINS
If both halves agree on MODEL_PATH, the server loads exactly the file train.py saved.

Every setting has a working default, so the app runs with no `.env` at all.
Copy `.env.example` to `.env` to override anything locally.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Load `backend_house_prediction/.env` if present. Real environment variables
# always win, so a container's settings are never overwritten by a stray file.
load_dotenv(ROOT / ".env", override=False)


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ---- Paths ----------------------------------------------------------------
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "pp-monthly-update-new-version.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "house_price_model.pkl"))

# ---- Server ---------------------------------------------------------------
# 0.0.0.0 by default so the server is reachable from outside any container
# (Render/Docker), no matter how the process is started. Local dev can still
# narrow it to 127.0.0.1 via `.env` or the API_HOST env var.
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
RELOAD = _get_bool("RELOAD", False)

# ---- CORS -----------------------------------------------------------------
# The React dev server and the Vite preview server, by default. Set
# ALLOWED_ORIGINS to a comma-separated list for deployment. "*" is accepted but
# is incompatible with credentialed requests, so it is not the default.
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

# ---- Training -------------------------------------------------------------
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))


def summary() -> dict:
    """Non-secret settings, surfaced on /health for debugging a deployment."""
    return {
        "model_path": str(MODEL_PATH),
        "allowed_origins": ALLOWED_ORIGINS,
        "host": API_HOST,
        "port": API_PORT,
    }
