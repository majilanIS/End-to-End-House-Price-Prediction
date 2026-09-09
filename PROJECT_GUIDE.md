# House Price Prediction — Full Project Guide

From raw data to a deployed web app. This guide walks through every layer of the project,
with extra depth on the **models** and **FastAPI**, and a full tour of the **deployment**.

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)
2. [System architecture](#2-system-architecture)
3. [Data understanding](#3-data-understanding)
4. [Data cleaning](#4-data-cleaning)
5. [Feature engineering](#5-feature-engineering)
6. [The models — deep dive](#6-the-models--deep-dive)
   - 6.1 What a regression model does
   - 6.2 Why we train on `log1p(price)`
   - 6.3 Models compared
   - 6.4 The winning model: HistGradientBoosting
   - 6.5 The preprocessing pipeline
   - 6.6 Model evaluation metrics
   - 6.7 Cross-validation
   - 6.8 The model bundle (what gets shipped to production)
   - 6.9 Why we refit on 100% of the data
7. [FastAPI — deep dive](#7-fastapi--deep-dive)
   - 7.1 What is an API?
   - 7.2 REST API vs. FastAPI
   - 7.3 What FastAPI gives us (and why we chose it)
   - 7.4 FastAPI vs. Flask vs. Django REST
   - 7.5 Every endpoint in this app
   - 7.6 How a request flows through the backend
   - 7.7 Pydantic validation
   - 7.8 CORS explained
8. [The backend folder, file by file](#8-the-backend-folder-file-by-file)
9. [Shared preprocessing: training = serving](#9-shared-preprocessing-training--serving)
10. [Prediction intervals and the location lookup](#10-prediction-intervals-and-the-location-lookup)
11. [The frontend (React + Vite)](#11-the-frontend-react--vite)
12. [Docker & containerization](#12-docker--containerization)
13. [CI/CD with GitHub Actions](#13-cicd-with-github-actions)
14. [Deployment — full tour](#14-deployment--full-tour)
    - 14.1 Choices we made
    - 14.2 GitHub Container Registry (ghcr.io)
    - 14.3 Docker Hub
    - 14.4 GitHub Pages (frontend)
    - 14.5 Render (backend)
    - 14.6 How the pieces talk to each other
15. [The complete flow — one read-through](#15-the-complete-flow--one-read-through)
16. [Limitations & honest caveats](#16-limitations--honest-caveats)

---

## 1. What this project does

A user enters a UK **postcode**, **property type**, **age**, **tenure**, **month** and
**year**. The system predicts that property's **sale price in GBP**, plus a rough
confidence range.

```
Postcode NW10 0DY   →   Predicted price: £367,209   (range £260k–£518k)
```

It is built on real UK housing market data (HM Land Registry **Price Paid Data**) and
served as a web app: a **FastAPI** backend exposes the model, a **React** frontend renders
the form and the result.

---

## 2. System architecture

```
┌────────────────────────────┐         ┌────────────────────────────┐
│        FRONTEND            │  HTTP   │         BACKEND            │
│   React   +   Vite (SPA)   │ ──────► │   FastAPI  +  uvicorn      │
│   GitHub Pages             │  fetch  │   Docker container         │
└────────────────────────────┘         └────────────┬───────────────┘
                                                    │ loads (once)
                                          ┌─────────▼─────────┐
                                          │  Model bundle     │
                                          │  house_price.pkl  │
                                          └───────────────────┘
```

Data flow, at a glance:

1. **Notebook** (`notebooks/house_price_prediction.ipynb`) — explore the raw CSV,
   justify cleaning decisions, compare models.
2. **`src/train.py`** — the production training script. Loads the CSV, builds features,
   trains + validates, and saves a **model bundle** (pickle) plus `metrics.json`.
3. **`app/main.py`** — a FastAPI app that loads that bundle and serves predictions over HTTP.
4. **Frontend** — a React SPA that calls the API and shows the estimate.
5. **CI/CD** — GitHub Actions builds and deploys both halves automatically.

---

## 3. Data understanding

### Source
HM Land Registry **Price Paid Data** — every residential property sale in England and
Wales, updated monthly. The project uses a single monthly **full extract** (~81k rows
after cleaning).

### Raw file format
The CSV ships **without a header row**. The 16 columns are defined in
`src/preprocessing.py::RAW_COLUMNS`:

| Column | Meaning |
|---|---|
| `transaction_id` | unique sale identifier |
| `price` | price in GBP |
| `date_of_transfer` | sale date |
| `postcode` | the only location signal at this granularity |
| `property_type` | `D` detached, `S` semi, `T` terraced, `F` flat, `O` other |
| `old_new` | `Y` newly built, `N` established |
| `duration` | `F` freehold, `L` leasehold |
| `paon` / `saon` | building / sub-building number or name |
| `street`, `locality` | address lines |
| `town_city`, `district`, `county` | named locations |
| `ppd_category` | `A` full market value · `B` not full market value |
| `record_status` | `A` added, `D` deletion record, `C` change |

### Key insights found during exploration
- `price` is **heavily right-skewed** — a few multi-million pound "sales" dominate the
  squared error of any model trained on raw GBP.
- `ppd_category == 'B'` rows are **not genuine market prices** (repossessions, buy-to-let
  portfolio transfers, transfers between related parties). About 19% of the file. Removing
  them is the single biggest score improvement in the project.
- Duplicate rows exist (the file includes `record_status == 'D'` deletions).
- Missing values appear mainly in **optional address fields** — but a missing postcode
  means no location signal at all, so those rows are dropped.

---

## 4. Data cleaning

`clean()` in `src/preprocessing.py` applies, in order:

1. **Drop rows with no postcode** — no location signal → not modellable.
2. **Drop `record_status == 'D'`** — deletion records correct earlier entries; they are
   not sales.
3. **Keep only `ppd_category == 'A'`** — real full-market-value transactions.
4. **Trim price to £20,000–£5,000,000** — removes ~99 bulk/commercial outliers. This
   alone lifts GBP-scale R² from 0.25 to 0.58, because otherwise ~99 extreme rows dominate
   the squared error.

These are **data decisions, not model tuning** — and they produced nearly all the
project's accuracy.

---

## 5. Feature engineering

`add_features()` derives the model inputs:

| Feature | Type | How it's built |
|---|---|---|
| `year`, `month` | numeric | parsed from `date_of_transfer` |
| `property_type`, `old_new`, `duration` | low-cardinality categorical | one-hot encoded |
| `town_city`, `district`, `county` | high-cardinality location | target-encoded |
| `area`, `area_code`, `sector` | postcode-derived location | target-encoded |

**Postcode → three nested levels** (called `area_code`, `area`, `sector`):

```
"NW10 0DY"
   ├─ area       "NW"        (2-letter postcode area)
   ├─ area_code  "NW10"      (outward code)
   └─ sector     "NW10 0"    (sector within the outward code)
```

Nesting matters: if the model has never seen a fine-grained code, it can still fall back
to a coarser geographic level instead of guessing.

**Why not one-hot the locations?** `town_city`, `district`, `county` etc. would explode
to ~989 sparse columns, train slowly, and score ~0.18 R² *worse*. Target encoding turns
each category into **one dense column** = the average target (price) for that category,
shrunk toward the global mean.

---

## 6. The models — deep dive

### 6.1 What a regression model does

This is a **regression** problem: predict a **continuous number** (price), not a
class/category. Each model is a function

```
price ≈ f(property_type, old_new, duration, year, month, town_city, district, county,
          area, area_code, sector)
```

The model learns `f` from ~64,850 training examples (80% of 81,066) and is judged on
~16,216 held-out examples (20%) it has never seen.

### 6.2 Why we train on `log1p(price)`

`price` is right-skewed: most houses cost £150k–£400k, a few cost millions. If you train
on raw GBP, the squared-error cost function obsesses over the expensive outliers.

Instead we train on **`log1p(price)` = ln(1 + price)`**:

- The target becomes roughly **normally distributed** (what linear/boosting models expect).
- An error of 10% on a £100k house and 10% on a £2M house become the *same size* error.
- Predictions are mapped back with `expm1(...)`, so the user still sees GBP.

```
log1p:  price  ──►  log(1 + price)     (train)
expm1:  log    ──►  exp(log) - 1        (predict back to GBP)
```

### 6.3 Models compared

`train.py` fits three candidates on the same split:

| Model | What it is | Why it's here |
|---|---|---|
| **Ridge** | linear regression with L2 penalty | cheap baseline, tests whether the problem is linear |
| **RandomForest** | many deep decision trees, averaged | captures non-linear interactions, no scaling needed |
| **HistGradientBoosting** | gradient-boosted trees built on histograms (Gradually-tuned bins) | fast, strong on tabular data, handles categoricals + scale well |

Result table (from `metrics.json`, log scale):

| Model | R² (log) | MdAPE | Overfit gap | Fit time |
|---|---|---|---|---|
| **HistGradientBoosting** | **0.700** | **18.6%** | 0.039 | 8s |
| RandomForest | 0.693 | 18.9% | 0.068 | 35s |
| Ridge | 0.640 | 20.8% | 0.031 | <1s |

**HistGradientBoosting wins** — and it's also the fastest to train.

### 6.4 The winning model: HistGradientBoosting

Gradient boosting builds a forest of **small decision trees in sequence**, where each new
tree tries to correct the errors of all the trees before it.

```
Prediction = y_mean
           + tree1.error_correction( ... )
           + tree2.error_correction( ... )
           + tree3.error_correction( ... )
           + ...
```

- **"Histogram"**: instead of evaluating every possible split point, it buckets features
  into a fixed number of bins (histograms) and finds splits on the bins. Much faster for
  large data than classic GBMs like `GradientBoostingRegressor`.
- **Hyperparameters used** (from `train.py`):

| Parameter | Value | Effect |
|---|---|---|
| `max_iter=800` | 800 boosting rounds | model capacity |
| `learning_rate=0.06` | small steps per tree | prevents overshooting |
| `max_leaf_nodes=63` | tree size cap | prevents overfitting |
| `min_samples_leaf=20` | each leaf ≥ 20 samples | smooths predictions |
| `l2_regularization=1.0` | ridge shrinkage per leaf | extra smoothing |
| `early_stopping=True` | stop if no gain on 10% validation | avoids wasted rounds |

### 6.5 The preprocessing pipeline

A scikit-learn **`Pipeline`** chains transformation + model so everything stays consistent:

```
Pipeline(
    preprocessor = ColumnTransformer(
        num   → StandardScaler        on [year, month]
        low   → OneHotEncoder          on [property_type, old_new, duration]
        high  → TargetEncoder          on [town_city, district, county, area, area_code, sector]
    ),
    regressor = HistGradientBoostingRegressor(...)
)
```

Crucially, the pipeline (including the encoders) is **fitted only on the training split**,
never on the test split — otherwise the test score would be dishonest (data leakage).

### 6.6 Model evaluation metrics

| Metric | What it measures | Value |
|---|---|---|
| **R² (log)** | fraction of variance explained | **0.700** (hold-out) |
| **RMSE (log)** | typical error magnitude, log scale | 0.343 |
| **MAE (log)** | average absolute error, log scale | 0.253 |
| **R² (GBP)** | variance explained after mapping back | 0.584 |
| **MAE (GBP)** | average £ error | £96,969 |
| **MdAPE** | **median** absolute percentage error | **18.5%** |
| **within 10% / 20%** | share of predictions that close | 29% / 53% |
| **overfit_gap** | train R² − test R² (lower = safer) | 0.042 |

The headline numbers: **half of all predictions land within ~18.5% of the true price**,
and 70% of price variance is explained on the log scale.

### 6.7 Cross-validation

A single train/test split can be a lucky split, so the project also runs **5-fold
cross-validation**:

```
[0.700, 0.696, 0.689, 0.684, 0.696]  →  0.693 ± 0.006
```

The hold-out R² (0.700) sits within one standard deviation of the CV mean, which confirms
it is **not luck**. `train.py` also checks "hold-out vs. CV drift" and reports
`STABLE / UNSTABLE`.

### 6.8 The model bundle (what ships to production)

`train.py` does **not** just dump a model. It saves a **bundle** (a dict) via `joblib`:

```python
{
  "model": Pipeline(...),            # the fitted pipeline
  "model_name": "HistGradientBoosting",
  "feature_columns": [...],           # the 11 columns the model expects
  "trained_at": ISO timestamp,
  "n_rows": 81066,
  "metrics": {...},                   # hold-out scores
  "cross_validation": {...},          # CV scores
  "location_lookup": {...},           # postcode area → town/district/county (2,246 areas)
  "log_residual_std": 0.3435,        # used for the price interval
}
```

This is what makes the API self-contained: the model, its inputs, its validation scores,
and its location map all travel together.

### 6.9 Why we refit on 100% of the data

The hold-out split existed **only to measure quality**. The model that ships is refit on
**all 81,066 rows**, so the deployed model uses every scrap of signal available. The
metrics we quote still come from the honest hold-out split — they describe the *family*
of model, not a retrained-on-test cheat.

---

## 7. FastAPI — deep dive

### 7.1 What is an API?

An **API** (Application Programming Interface) is a contract: a set of rules for one
program to talk to another. A **web API** exposes URLs; clients send HTTP requests and
get JSON back.

Here, the frontend (browser) sends:
```
POST /predict   {"postcode": "NW10 0DY", "property_type": "F", ...}
```
and the backend answers:
```
{"predicted_price": 367209.44, "lower_bound": ..., "upper_bound": ...}
```

### 7.2 REST API vs. FastAPI

This is an important distinction — they are **different categories**:

| REST | FastAPI |
|---|---|
| An **architectural style** (rules/constraints): resources, HTTP verbs, statelessness, status codes | A **Python framework** (a tool) for building web APIs |
| Not a technology — any stack can be "RESTful" | Built on Starlette (ASGI) + Pydantic |
| Defines **how** endpoints should behave (GET reads, POST creates, PUT updates, DELETE removes; resources named by URL) | Gives you the **fastest possible way** to implement such endpoints in Python |

In short: **REST is the design philosophy, FastAPI is the implementation**. Our API *is*
RESTful — every route acts on a resource (`/predict`, `/location/{postcode}`), uses the
correct verbs (`GET` for reads, `POST` for predictions), and uses HTTP status codes
(`200`, `400`, `404`, `413`, `503`).

### 7.3 What FastAPI gives us (and why we chose it)

1. **Request validation for free** — Pydantic models validate/inspect the JSON body,
   returning a precise `422` with field-level errors when input is wrong.
2. **Automatic interactive docs** — `http://localhost:8000/docs` (Swagger) and `/redoc`.
   Every endpoint, schema, and example is documented from the code, no extra work.
3. **Type hints drive everything** — `def predict(req: PredictionRequest) -> PredictionResponse:`
   means the response is serialized and documented automatically.
4. **Performance** — built on Starlette and ASGI (async), it is one of the fastest
   Python web frameworks (comparable to Node/Go in benchmarks).
5. **Modern async support** — `async def` endpoints for I/O-heavy code.
6. **Clean structure** — decorators map functions to URLs, so the code reads top-to-bottom.

### 7.4 FastAPI vs. Flask vs. Django REST

| | FastAPI | Flask | Django REST Framework |
|---|---|---|---|
| Validates request/response automatically | **Yes (Pydantic)** | No — you hand-validate | Partially (serializers) |
| Auto interactive docs | **Yes, free** | No (add `flasgger`) | With `drf-yasg` extra |
| Async support | **Native (ASGI)** | WSGI, awkward | Via ASGI mode |
| Type-hint driven | **Yes** | No | No |
| Best for | **ML/data APIs, microservices** | Small/quick services | Full Django apps with ORM+admin |

For an ML project — where input validation and auto-docs matter and there is no ORM/admin
need — **FastAPI is the natural fit**.

### 7.5 Every endpoint in this app

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | service banner + links |
| `GET` | `/health` | is the model loaded, what config is active |
| `GET` | `/schema` | the valid category codes (drives the frontend form) |
| `GET` | `/metrics` | the deployed model's hold-out + CV scores |
| `GET` | `/location/{postcode}` | resolve postcode → town/district/county |
| `POST` | `/predict` | price + range for one property |
| `POST` | `/predict/batch` | up to 500 properties per call |

### 7.6 How a request flows through the backend

```
Browser (React)                     FastAPI                         Python modules
─────────────────                ────────────────                ─────────────────
POST /predict  ──────────────►   cors middleware        ──►  app/main.py
   JSON body                     pydantic validates            predict_one()
                                 (422 if invalid)  ─────────►  src/predict.py
                                                              src/preprocessing.py
                                                              (build features, run model)
                                 200 OK + typed response  ◄───────┘
   "£367,209"  ◄───────────── JSON
```

1. **Middleware order**: CORS middleware runs first, letting allowed browser origins
   through.
2. **Validation**: pydantic parses the JSON body into `PredictionRequest` — wrong types,
   invalid postcodes, or out-of-range years fail here with a detailed `422`.
3. **Business logic**: `predict_one()` resolves location from the postcode if needed,
   builds a one-row feature frame, runs the bundle's pipeline, and maps the log price
   back to GBP.
4. **Response**: the result is serialized as `PredictionResponse` (typed) and returned
   as JSON with status `200`.

### 7.7 Pydantic validation

Two examples from `app/main.py`:

```python
year: int = Field(..., ge=1995, le=2030)     # must be 1995..2030
postcode: str = Field(..., examples=["NW10 0DY"])

@field_validator("postcode")
def postcode_must_have_outward_and_inward(cls, value):
    if " " not in value.upper().strip():
        raise ValueError("postcode must include the space, e.g. 'NW10 0DY'")
    return value
```

The `Literal` types pin the categorical codes to exactly `D/S/T/F/O`, `Y/N`, `F/L` — a
request with `"property_type": "villa"` is rejected by Pydantic, before our code even runs.

### 7.8 CORS explained

The frontend lives on one origin (e.g. `https://majilanis.github.io`) and the API on
another (e.g. `https://house-price-api.onrender.com`). Browsers block cross-origin
requests unless the API **opt in** with a `Access-Control-Allow-Origin` header.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,   # explicit list, not "*"
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

`ALLOWED_ORIGINS` comes from the environment (`.env` / Render env vars), defaulting to the
local dev servers. For production we set it to `https://majilanis.github.io`. Note: we
list origins **explicitly** rather than `"*"` because `*` is incompatible with
credentials (`cookies`/`authorization`) and is a security footgun.

---

## 8. The backend folder, file by file

```
backend_house_prediction/
├── notebooks/house_price_prediction.ipynb   # EDA, model comparison, API + Docker demo
├── src/
│   ├── config.py          # settings from .env / environment variables
│   ├── preprocessing.py   # loading, cleaning, features — shared by training AND API
│   ├── train.py           # trains, validates, cross-checks, writes the bundle
│   └── predict.py         # loads the bundle; predict_one(), resolve_location()
├── app/
│   └── main.py            # the FastAPI app (all endpoints)
├── models/
│   ├── house_price_model.pkl   # joblib bundle (git-ignored)
│   └── metrics.json            # saved scores of the shipped model
├── data/                       # raw + cleaned CSVs (git-ignored, fetched per README)
├── requirements.txt            # runtime deps for training + serving
├── requirements-runtime.txt    # (older layering; requirements.txt is the single source)
├── .env / .env.example         # config
├── Dockerfile
└── .dockerignore
```

The important architectural decision: **training and serving share `preprocessing.py`**.
That single file is imported by the notebook, by `train.py`, *and* by the FastAPI app.
The features built at serving time are therefore byte-for-byte identical to the ones used
at training time — which removes the classic "works in the notebook, wrong in production"
failure mode.

---

## 9. Shared preprocessing: training = serving

`src/preprocessing.py` is the single source of truth:

```
Notebook ──┐
train.py ──┼──►  add_features()   clean()   load_raw()   build_preprocessor()
FastAPI  ──┘
```

In production, a request arrives with just a few fields. `features_from_request()` builds
the *same* 11 columns that training produced, so the pipeline's `ColumnTransformer`
receives exactly what it expects.

```python
# predict.py — one row, same features as training
features = features_from_request(postcode=..., property_type=..., ...)
price = np.expm1(bundle["model"].predict(features))[0]
```

---

## 10. Prediction intervals and the location lookup

### The confidence range
The model trains in log space; its residuals are roughly normal. The bundle stores
`log_residual_std ≈ 0.3435`. A ±1σ band in log space becomes an **asymmetric** band in GBP:

```
log bound = log(price) ± 0.3435
price_low  = expm1(log(price) - 0.3435)   # narrower below
price_high = expm1(log(price) + 0.3435)   # wider above
```

Asymmetry matches reality: over-estimates and under-estimates are not symmetric in money.

### The location lookup
`town_city`, `district`, `county` are **target-encoded**, so they must match the training
vocabulary character-for-character. If a user typed `"Gtr London"` instead of
`"GREATER LONDON"`, the encoder would silently fall back to the global mean — a
*confidently wrong* answer.

Fix: during training we build a **postcode area → location map** (2,246 areas) inside the
bundle. The API resolves location from the postcode itself:

```
GET /location/NW10 0DY
→ { "outward_code": "NW10", "town_city": "LONDON", "district": "BRENT",
    "county": "GREATER LONDON", "n_sales": 8123, "confidence": 0.94 }
```

`resolve_location()` (in `predict.py`) looks up the outward code, and `predict_one()`
uses those three fields only when the client did not supply better ones.

---

## 11. The frontend (React + Vite)

- **React** renders a form and results; **Vite** builds/bundles the SPA.
- `src/api.js` wraps `fetch` and points at the backend. When `VITE_API_BASE_URL` is empty,
  requests use **relative paths** (dev proxy); when set, they go to that origin directly.
- `App.jsx` behaviour:
  - Loads `/schema` and `/metrics` on mount to build the form and stats panels.
  - **Postcode autolookup** (debounced 350ms) calls `/location/{postcode}`; if the area is
    unknown it reveals manual town/district/county inputs instead.
  - Submits to `/predict`, renders `£ price`, the ±1σ range bar, model name, location
    source, and the typical-error disclaimer.
  - Light/dark theme persisted to `localStorage`.
- It is a **static SPA** — pure HTML/JS/CSS after build, which is why it deploys to
  **GitHub Pages** with zero server code.

---

## 12. Docker & containerization

### What Docker gives us
**Docker** packages the backend (Python + model + code + system deps) into one **image**,
so it runs identically anywhere Docker exists — your laptop, GitHub Actions, Render.

### The backend Dockerfile (`backend_house_prediction/Dockerfile`)

```dockerfile
FROM python:3.11-slim          # small, lean base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # installs pandas, sklearn, fastapi...

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/          # the trained bundle ships INSIDE the image

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser                    # run as non-root (security best practice)

EXPOSE 8000

HEALTHCHECK CMD python -c "...check /health..."

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Why it's small: only **runtime** dependencies are installed (no jupyter/matplotlib), and
the image just copies in the already-trained model — the container **serves**, it doesn't
train.

### Building & running locally

```bash
docker build -t house-price-api ./backend_house_prediction
docker run -p 8000:8000 house-price-api
curl http://localhost:8000/health
```

### Why the frontend has no container anymore
The SPA is static files. Containerizing it (nginx) was overkill, so we removed its
Dockerfile and deploy it directly to GitHub Pages instead. **Only the backend is a
container** — exactly as requested.

---

## 13. CI/CD with GitHub Actions

`.github/workflows/` contains two workflows that trigger automatically on push to `main`
(or manually via **Actions → Run workflow**):

### `backend-docker.yml` — build & push the backend image
```
1. checkout the repo
2. set up buildx + login to ghcr.io (GitHub's container registry)
3. docker metadata → tags (branch, semver, sha)
4. docker buildx build … push to ghcr.io/majilanis/end-to-end-house-price-prediction:main
```

Uses `GITHUB_TOKEN` — **no secrets needed**.

### `frontend-pages.yml` — build & deploy the React app
```
1. checkout, setup-node, npm ci
2. npm run lint              <- code quality gate
3. npm run build             <- Vite produces dist/
4. upload-pages-artifact     <- save dist/ as an artifact
5. deploy-pages              <- publish to GitHub Pages
```

Also fully automatic with `GITHUB_TOKEN`.

Both are triggered on push to `main`, restricted to paths that actually changed
(`backend_house_prediction/**`, `frontend_house_prediction/**`).

---

## 14. Deployment — full tour

### 14.1 Choices we made

| Piece | Where | Why |
|---|---|---|
| Backend image | **ghcr.io** + **Docker Hub** | registries; ghcr is private until made public, Docker Hub is public and directly pullable by Render |
| Frontend | **GitHub Pages** | static SPA, zero cost, zero server |
| Backend runtime | **Render** | free-tier container hosting that pulls a Docker image |
| CI/CD | **GitHub Actions** | already in the repo, uses `GITHUB_TOKEN` |

### 14.2 GitHub Container Registry (ghcr.io)
`docker/login-action@v3` authenticates to `ghcr.io` with the `GITHUB_TOKEN` (no manual
setup). Image:
```
ghcr.io/majilanis/end-to-end-house-price-prediction:main
```
*Note:* a repo-scoped ghcr package is **private** by default; to let Render pull it you'd
either make it public (repo → Packages → Package settings → Change visibility) or provide
registry credentials.

### 14.3 Docker Hub
Because you already have a Docker Hub account (`chekoledocker`), we also built and pushed
publicly:
```
docker build -t chekoledocker/house-price-api:latest ./backend_house_prediction
docker push chekoledocker/house-price-api:latest
```
This image is **public** — Render can pull it with no credentials. This is the image the
Render service actually uses.

### 14.4 GitHub Pages (frontend)
Deployed by `frontend-pages.yml`. Live at:
```
https://majilanis.github.io/End-to-End-House-Price-Prediction/
```
Because the site lives under a sub-path (`/End-to-End-House-Price-Prediction/`), the
build sets Vite's `base` via `BASE_PATH`:

```yaml
- run: npm run build
  env:
    BASE_PATH: /${{ github.event.repository.name }}/
```
and `vite.config.js` reads it: `base: process.env.BASE_PATH || '/'`.

### 14.5 Render (backend)
Two ways to deploy:

**(a) Blueprint (`render.yaml`)** — already in the repo:
```yaml
services:
  - type: web
    name: house-price-api
    runtime: image
    image:
      url: docker.io/chekoledocker/house-price-api:latest
    envVars:
      - key: API_HOST      value: 0.0.0.0
      - key: API_PORT      value: 8000
      - key: RELOAD        value: "false"
      - key: ALLOWED_ORIGINS  value: https://majilanis.github.io
    healthCheckPath: /health
```
On Render: **New → Blueprint → select the GitHub repo** → Apply. Render reads
`render.yaml` and provisions the service.

**(b) Manual UI** — **New → Web Service → Deploy from Docker Image**:
- Image URL: `chekoledocker/house-price-api:latest`
- Health check path: `/health`
- Env vars: as in the blueprint.
- **Leave the start command empty** — the Dockerfile's `CMD` already runs
  `uvicorn app.main:app ...`.

The service exposes on the external port Render assigns (commonly `10000`), but the
container listens on **8000** internally; Render's platform routes traffic for you.

### 14.6 How the pieces talk to each other

```
Browser
  │
  ▼
GitHub Pages (React SPA)  ──fetch /predict──►  Render (FastAPI container)
  https://majilanis.github.io/...              https://house-price-api.onrender.com
                                                    │
                                                    ▼
                                             model bundle inside the image
```

The browser makes a cross-origin `POST /predict` to Render. Render's CORS middleware
allows the origin `https://majilanis.github.io` (set in `ALLOWED_ORIGINS`), the request
is validated by Pydantic, the model runs, and the JSON estimate returns to the page.

For that to work end-to-end, remember to set the **frontend's** API base URL during its
build to the Render URL. In the Vercel/GitHub-Pages deployment, `VITE_API_BASE_URL` must
equal `https://house-price-api.onrender.com` (without trailing slash).

---

## 15. The complete flow — one read-through

1. **Fetch data** — download HM Land Registry Price Paid CSV into `backend_house_prediction/data/`.
2. **Explore** — notebook: inspect columns, missing values, skew, the `ppd_category`
   problem.
3. **Clean + engineer** — `preprocessing.py`: drop non-market sales, trim price range,
   derive year/month, split postcode into area/area_code/sector.
4. **Model** — `train.py`: compare Ridge / RandomForest / HistGradientBoosting on
   `log1p(price)`, evaluate on hold-out, run 5-fold CV, refit winner on all data.
5. **Bundle** — save pipeline + metrics + location_lookup to `models/house_price_model.pkl`.
6. **Serve** — FastAPI loads the bundle once at startup, exposes `/predict`, `/health`,
   `/metrics`, `/schema`, `/location`.
7. **Frontend** — React builds a form, autolooks-up location, posts to `/predict`,
   shows the price + range.
8. **Containerize** — Dockerfile packages backend + model; image pushed to ghcr.io and
   Docker Hub.
9. **Automate** — GitHub Actions builds/deploys frontend to Pages and backend to a
   registry on every push.
10. **Host** — Render runs the Docker image as a web service; GitHub Pages hosts the SPA;
    CORS wires them together.

---

## 16. Limitations & honest caveats

- **The ceiling is the data, not the model.** Price Paid Data records *what sold for how
  much* — not *what was sold*. There is **no floor area, bedrooms, garden, or condition**.
  Two houses with identical postcode/type/tenure can legitimately differ by ~50%.
- ~**96% of the current file is 2025–2026**, so `year` carries almost no trend signal here;
  on a decade-long extract it would matter more.
- Median error is ~18.5%; worst at the extreme ends of the market (cheapest/most expensive
  deciles), where the missing attributes matter most.
- Crossing R² ≈ 0.70 requires joining property attributes (e.g. the **EPC register** for
  floor area and room counts), not more model tuning.
- Predictions for postcodes **not in the training data** fall back to the encoder's global
  mean (`location_source: "unknown_postcode"`) — usable, but treat with extra care.