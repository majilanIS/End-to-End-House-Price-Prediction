"""FastAPI service exposing the trained house price model.

Run locally:
    uvicorn app.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src import config
from src.predict import load_bundle, predict_one, resolve_location
from src.preprocessing import DURATION, OLD_NEW, PROPERTY_TYPES, outward_code

# Loaded once at startup so the first request does not pay the unpickling cost.
_startup_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _startup_error
    try:
        load_bundle()
    except FileNotFoundError as exc:  # keep the app up so /health can report it
        _startup_error = str(exc)
    yield


app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predicts UK residential sale prices from HM Land Registry Price Paid data. "
        "Trained on full-market-value transactions only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The React frontend runs on a different origin from the API, so the browser
# blocks its requests unless the API opts in. Origins are listed explicitly
# rather than using "*" — set ALLOWED_ORIGINS in .env for deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    postcode: str = Field(..., examples=["NW10 0DY"], description="Full UK postcode")
    property_type: Literal["D", "S", "T", "F", "O"] = Field(
        ..., description="D=Detached, S=Semi-detached, T=Terraced, F=Flat, O=Other"
    )
    old_new: Literal["Y", "N"] = Field(..., description="Y=new build, N=established")
    duration: Literal["F", "L"] = Field(..., description="F=Freehold, L=Leasehold")
    year: int = Field(..., ge=1995, le=2030, description="Year of sale")
    month: int = Field(..., ge=1, le=12, description="Month of sale")

    # Optional: resolved from the postcode when omitted. Supplying them by hand is
    # supported but risky — they must match the training vocabulary exactly.
    town_city: str | None = Field(None, examples=["LONDON"])
    district: str | None = Field(None, examples=["BRENT"])
    county: str | None = Field(None, examples=["GREATER LONDON"])

    @field_validator("postcode")
    @classmethod
    def postcode_must_have_outward_and_inward(cls, value: str) -> str:
        cleaned = " ".join(value.strip().upper().split())
        if " " not in cleaned:
            raise ValueError("postcode must include the space, e.g. 'NW10 0DY'")
        return cleaned

    @field_validator("town_city", "district", "county")
    @classmethod
    def normalise_place(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned or None


class PredictionResponse(BaseModel):
    predicted_price: float
    lower_bound: float
    upper_bound: float
    currency: str = "GBP"
    model_name: str
    # How town/district/county were determined: "postcode_lookup", "provided",
    # or "unknown_postcode" (outside the training data — treat the estimate with care).
    location_source: str
    town_city: str
    district: str
    county: str


class LocationResponse(BaseModel):
    outward_code: str
    town_city: str
    district: str
    county: str
    n_sales: int
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    trained_at: str | None = None
    n_training_rows: int | None = None
    known_postcode_areas: int | None = None
    config: dict | None = None
    detail: str | None = None


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "House Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    if _startup_error is not None:
        return HealthResponse(status="degraded", model_loaded=False, detail=_startup_error)
    bundle = load_bundle()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=bundle["model_name"],
        trained_at=bundle["trained_at"],
        n_training_rows=bundle["n_rows"],
        known_postcode_areas=len(bundle.get("location_lookup") or {}),
        config=config.summary(),
    )


@app.get("/schema", tags=["meta"])
def schema() -> dict:
    """The accepted category codes, so a client can build a form without guessing."""
    return {"property_type": PROPERTY_TYPES, "old_new": OLD_NEW, "duration": DURATION}


@app.get("/metrics", tags=["meta"])
def metrics() -> dict:
    """Quality of the deployed model, as measured on the hold-out split."""
    if _startup_error is not None:
        raise HTTPException(status_code=503, detail=_startup_error)
    bundle = load_bundle()
    return {
        "model_name": bundle["model_name"],
        "trained_at": bundle["trained_at"],
        "holdout": bundle["metrics"],
        "cross_validation": bundle["cross_validation"],
    }


@app.get("/location/{postcode}", response_model=LocationResponse, tags=["prediction"])
def location(postcode: str) -> LocationResponse:
    """Resolve a postcode to the town / district / county the model was trained on.

    Lets a client fill those fields automatically instead of asking a user to type
    names that must match the training vocabulary character for character.
    """
    if _startup_error is not None:
        raise HTTPException(status_code=503, detail=_startup_error)
    resolved = resolve_location(postcode)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"No sales on record for postcode area "
                   f"'{outward_code(postcode)}'. Enter the location manually.",
        )
    return LocationResponse(**resolved)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    if _startup_error is not None:
        raise HTTPException(status_code=503, detail=_startup_error)
    try:
        result = predict_one(**request.model_dump())
    except Exception as exc:  # surface as a 400 rather than an opaque 500
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["prediction"])
def predict_batch(requests: list[PredictionRequest]) -> list[PredictionResponse]:
    if _startup_error is not None:
        raise HTTPException(status_code=503, detail=_startup_error)
    if len(requests) > 500:
        raise HTTPException(status_code=413, detail="Send at most 500 properties per call.")
    return [predict(item) for item in requests]


# Makes `python -m app.main` work, so there is no need to remember the uvicorn
# invocation. `uvicorn app.main:app --reload` remains equivalent.
if __name__ == "__main__":
    import uvicorn

    print(f"House Price API -> http://{config.API_HOST}:{config.API_PORT}/docs")
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.RELOAD,
    )
