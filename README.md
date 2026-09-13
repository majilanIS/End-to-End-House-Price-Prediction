# House Price Prediction — End to End

Predicts UK residential sale prices from [HM Land Registry Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads),
served through a **FastAPI backend** (shipped as a Docker image) with a **React frontend** (deployed to static hosting).

| | |
|---|---|
| **Model** | HistGradientBoostingRegressor on `log1p(price)` |
| **R² (log scale)** | 0.700 hold-out · **0.693 ± 0.006** 5-fold cross-validated |
| **Median absolute error** | 18.6% |
| **Training rows** | 81,066 full-market-value sales |

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │           GitHub Pages (static SPA)          │
                    │   https://<user>.github.io/<repo>/            │
                    │   built by .github/workflows/frontend-pages … │
                    └──────────────────────┬───────────────────────┘
                                           │  HTTPS
                                           │  CORS allow-list checked
                    ┌──────────────────────▼───────────────────────┐
                    │     Render service – Docker image backend    │
                    │   https://house-price-api-latest-1.onrender… │
                    │   FastAPI  (uvicorn, port 8000)              │
                    └───────────────┬──────────────┬───────────────┘
                                    │              │
                             ┌──────▼─────┐  ┌─────▼──────────┐
                             │  Model     │  │ Postcode →     │
                             │ pipeline   │  │ location map   │
│ (joblib)   │  │ (2,246 areas) │
                             └────────────┘  └────────────────┘
```

Two independent deploys, one contract (the HTTP API):

| Piece | Where it runs | Built by | Public URL |
|---|---|---|---|
| **Frontend** (React SPA) | GitHub Pages | `frontend-pages.yml` (GitHub Actions) | `https://<user>.github.io/<repo>/` |
| **Backend** (FastAPI + model) | Render, running `chekoledocker/house-price-api` image | `backend-docker.yml` or `docker build` / Docker Hub | `https://house-price-api-latest-1.onrender.com` |

Both are configured to work together out of the box. The frontend is a **static bundle**: the API URL is baked into its JavaScript at build time (`VITE_API_BASE_URL`) — never put secrets there.

---

## Repository layout

```
.
├── backend_house_prediction/          # FastAPI service, training, model
│   ├── app/
│   │   └── main.py                    # FastAPI application
│   ├── src/
│   │   ├── config.py                  # settings from env / .env (shared)
│   │   ├── preprocessing.py           # cleaning + features (shared)
│   │   ├── train.py                   # trains, validates, writes the bundle
│   │   └── predict.py                 # loads the bundle, returns a price
│   ├── models/
│   │   ├── house_price_model.pkl      # joblib bundle (committed, ~1.5 MB)
│   │   └── metrics.json               # scores of the saved model
│   ├── data/                          # raw + cleaned CSVs (git-ignored)
│   ├── .env.example                   # copy to .env for local overrides
│   ├── requirements.txt
│   └── Dockerfile                     # ships src/ + app/ + models/
│
├── frontend_house_prediction/         # React + Vite SPA
│   ├── src/
│   │   ├── api.js                     # backend client (BASE_URL handling)
│   │   ├── App.jsx                    # form + result + theme toggle
│   │   └── App.css / index.css        # design tokens, light + dark
│   ├── .env.example
│   ├── package.json
│   └── vite.config.js                 # dev proxy → backend
│
├── .github/workflows/
│   ├── backend-docker.yml             # build + push backend image (Docker Hub)
│   └── frontend-pages.yml             # build + deploy frontend to GitHub Pages
│
├── docker-compose.yml                 # local backend with one command
├── render.yaml                        # Render blueprint (documentation/starter)
├── .env.example                       # vars for docker compose
└── README.md
```

`src/preprocessing.py` is deliberately shared: training and serving build features with the
same code, which is what stops the classic "works in the notebook, wrong in production" bug.

---

## Prerequisites

- **Python 3.11+** and `pip`
- **Node 20+** and `npm`
- **Docker** (optional — for the containerized backend)

No database, no external services required to run locally: the model and the postcode
lookup ship inside the repository.

---

## Quick start (local, without Docker)

### 1. Backend

```bash
cd backend_house_prediction
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)

pip install -r requirements.txt
python -m app.main                # serves the API on http://localhost:8000
```

Open **http://localhost:8000/docs** for an interactive API explorer, or smoke-test:

```bash
curl http://localhost:8000/health          # {"status":"ok","model_loaded":true,...}
```

Notes that trip people up:

- It is `python -m app.main` (a module path), never `main.py`.
- `uvicorn app.main:app --reload` is equivalent but only works while the venv is active.
- The venv ships **no training data** — a trained model is already committed, so you can
  serve immediately. To retrain, see [Retraining the model](#-retraining-the-model).

### 2. Frontend

```bash
cd frontend_house_prediction
npm install
npm run dev                        # http://localhost:5173
```

In development the SPA proxies API calls to `http://127.0.0.1:8000` (see `vite.config.js`),
so the browser stays on one origin and **CORS never applies**.

---

## Run the backend with Docker

```bash
cp .env.example .env               # optional; every value has a default
docker compose up --build -d
curl http://localhost:8000/health
```

`docker-compose.yml` runs only the **backend** (the frontend is static hosting — see
[Deploying](#deploying)). It mounts no model of its own: the Docker image copies
`models/` in at build time.

| Root `.env` var | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | public port of the API |
| `ALLOWED_ORIGINS` | `http://localhost:5173,https://majilanis.github.io` | CORS origins the backend allows |

---

## Deploying

> The trained model is committed to the repo (`models/house_price_model.pkl`, ~1.5 MB).
> That is deliberate: the training CSV is far too large for git, so a deploy host cannot
> retrain — it can only run what we ship. **The model and the code that loads it are
> versioned together.**

### Backend → Render (Docker image)

1. **Publish the image.** On `main`, `.github/workflows/backend-docker.yml` builds the
   image, smoke-tests `/health` + `/predict` inside the container, and pushes
   `chekoledocker/house-price-api:latest` to Docker Hub. It needs two repo secrets:
   - `DOCKERHUB_USERNAME` — your Docker Hub username
   - `DOCKERHUB_TOKEN` — an access token from <https://hub.docker.com/settings/security>

   (Or publish manually: `docker build -t chekoledocker/house-price-api ./backend_house_prediction && docker push chekoledocker/house-price-api`.)

2. **Create the Render service.** Render → *New → Web Service → Deploy an existing image*,
   image `docker.io/chekoledocker/house-price-api:latest`. Set the env vars below and
   health check path `/health`.

3. **Redeploy.** After every image push: *Manual Deploy → Clear build cache & deploy*.
   A cached Docker layer can keep serving a model file from an older build — clearing
   cache is the reliable way to avoid exactly that.

Backend env vars on Render:

| Variable | Value |
|---|---|
| `API_HOST` | `0.0.0.0` |
| `API_PORT` | `8000` |
| `RELOAD` | `false` |
| `ALLOWED_ORIGINS` | see below |

> `render.yaml` in this repo documents this setup, but its env values are `sync: false` —
> you must set `ALLOWED_ORIGINS` in the Render dashboard for it to take effect.

**`ALLOWED_ORIGINS` must list every origin that calls the API from a browser.** For the
frontends used in this project:

```
https://<user>.github.io,https://end-to-end-house-price-prediction-ten.vercel.app,http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173
```

### Frontend → GitHub Pages

1. Enable Pages at *repo → Settings → Pages* (Source: GitHub Actions).
2. On `main`, `.github/workflows/frontend-pages.yml`:
   - injects `BASE_PATH=/<repo>/` (Pages is served under a sub-path),
   - injects `VITE_API_BASE_URL=<backend-url>` (baked into the built JS),
   - uploads `frontend_house_prediction/dist` and deploys it.

The workflow's `VITE_API_BASE_URL` is the one and only place to point the frontend at a
backend. Change it, push, and Pages re-deploys.

---

## Verification checklist

After a deploy, hit the public backend `https://house-price-api-latest-1.onrender.com`:

| Endpoint | Method | Expect |
|---|---|---|
| `/health` | GET | `status:"ok"`, `model_loaded:true`, and your frontend origin inside `config.allowed_origins` |
| `/schema` | GET | `{property_type, old_new, duration}` code maps |
| `/metrics` | GET | hold-out + cross-validation scores of the deployed model |
| `/predict` | POST | a JSON price with `predicted_price` and a confidence band |
| `/predict/batch` | POST | array of up to 500 predictions |
| `/docs` | GET | Swagger UI loads |

```bash
curl -X POST https://house-price-api-latest-1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"postcode":"NW10 0DY","property_type":"F","old_new":"N","duration":"L","year":2026,"month":6}'
# {"predicted_price":379621.38,"lower_bound":269267.09,"upper_bound":535202.23,...}
```

Frontend checklist:

- Open the Pages URL and confirm the green "HistGradientBoosting" badge appears
  (it comes from `/health`).
- Submit the form — the estimate renders with `MdAPE` accuracy note.
- If the badge shows a warning or a JSON error, open DevTools → Network and confirm
  requests go to the Render URL (not relative paths) — a missing `VITE_API_BASE_URL`
  is almost always the cause.

---

## Configuration

Both halves read a `.env` file; both run without one. Copy the examples to start:

```bash
cp backend_house_prediction/.env.example  backend_house_prediction/.env
cp frontend_house_prediction/.env.example frontend_house_prediction/.env
cp .env.example .env                      # for docker compose (root)
```

**Backend** (`backend_house_prediction/.env`) — real environment variables always win, so a
container's settings are never overridden by a stray file.

| Variable | Default | Purpose |
|---|---|---|
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | where the server binds |
| `RELOAD` | `false` | auto-reload on code changes (development only) |
| `ALLOWED_ORIGINS` | Vite dev + preview | comma-separated CORS origins |
| `MODEL_PATH` | `models/house_price_model.pkl` | model bundle to serve |
| `DATA_PATH` | `data/pp-monthly-update-new-version.csv` | training data (retrain only) |
| `RANDOM_STATE` / `TEST_SIZE` / `CV_FOLDS` | `42` / `0.2` / `5` | training knobs |

**Frontend** (`frontend_house_prediction/.env`) — only `VITE_`-prefixed variables reach the
browser, and they are baked into the built JavaScript. Never put secrets here.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | *(empty)* | Empty → relative paths + the Vite dev proxy (browser stays on one origin, no CORS). Set it (e.g. `https://api.your-domain.com`) for a deployed build — then add that frontend's origin to the backend's `ALLOWED_ORIGINS`. |
| `VITE_DEV_API_TARGET` | `http://127.0.0.1:8000` | where the dev proxy forwards |

---

## Retraining the model

**Only do this if the data changed or the pinned scikit-learn version changed.**

```bash
cd backend_house_prediction
.venv\Scripts\activate
python -m src.train          # ~1 min; loads data/, compares models, writes the bundle
git add models/house_price_model.pkl models/metrics.json
```

That pkl `IS` your deployment. After committing, run the pipeline again
(CI → Docker Hub → Render), because the hosted backend cannot retrain itself.

> **Scikit-learn is pinned exactly** (`scikit-learn==1.9.0`). A pickled sklearn Pipeline is
> not portable across minor versions: a 1.8-trained model fails to load on 1.9 with the
> famously unhelpful `ModuleNotFoundError: No module named '_loss'`. The bundle records the
> version that built it and `load_bundle()` compares it at startup — a mismatch now reports
> the real cause instead. If the API ever says `degraded` on `/health` or dies with `_loss`:
> your deployed model predates your installed scikit-learn. Retrain and commit as above,
> then redeploy — and if Render still serves the old model, *Clear build cache & deploy*.

---

## Data

The dataset is not committed (17 MB, and it is refreshed monthly). Download the
**Price Paid Data monthly update** from HM Land Registry and save it as:

```
backend_house_prediction/data/pp-monthly-update-new-version.csv
```

It ships **without a header row**; the 16-column schema is defined in
`src/preprocessing.py::RAW_COLUMNS`.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | is the model loaded, when was it trained, active config |
| `GET` | `/schema` | valid category codes, so a client can build a form |
| `GET` | `/metrics` | hold-out and cross-validation scores of the deployed model |
| `GET` | `/location/{postcode}` | resolve a postcode to town / district / county |
| `POST` | `/predict` | price + confidence interval for one property |
| `POST` | `/predict/batch` | up to 500 properties per call |

Only the postcode and the property's own attributes are required — location is derived
server-side:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"postcode":"NW10 0DY","property_type":"F","old_new":"N",
       "duration":"L","year":2026,"month":6}'
```

```json
{
  "predicted_price": 379621.38,
  "lower_bound": 269267.09,
  "upper_bound": 535202.23,
  "currency": "GBP",
  "model_name": "HistGradientBoosting",
  "location_source": "postcode_lookup",
  "town_city": "LONDON",
  "district": "BRENT",
  "county": "GREATER LONDON"
}
```

**Why location is derived, not typed.** `town_city`, `district` and `county` are
target-encoded, so they must match the training vocabulary character for character. A user
typing `Gtr London` instead of `GREATER LONDON` would silently fall back to the encoder's
global mean and get a confidently wrong answer with a `200`. Training therefore ships a
postcode → location map (2,246 areas) inside the model bundle, and the API resolves those
fields itself. You may still pass them explicitly to override; `location_source` reports
which path was taken (`postcode_lookup`, `provided`, or `unknown_postcode`).

The bounds are ±1 residual sigma in log space, which makes the interval asymmetric in
GBP — matching how price uncertainty actually behaves.

**Field codes** — `property_type`: `D` detached, `S` semi-detached, `T` terraced,
`F` flat/maisonette, `O` other · `old_new`: `Y` new build, `N` established ·
`duration`: `F` freehold, `L` leasehold.

---

## How the model was built

### What actually moved the needle

| Change | R² (log) | R² (GBP) |
|---|---|---|
| Baseline: LinearRegression, one-hot location | 0.406 | 0.030 |
| Target-encode location instead of one-hot | 0.517 | 0.113 |
| Keep only `ppd_category == 'A'` (real market sales) | 0.697 | 0.252 |
| Trim price to £20k–£5M (removes 99 rows) | **0.700** | **0.583** |

Two data decisions, not model tuning, produced nearly all the gain:

1. **`ppd_category B` rows are not market prices.** 19% of the file is repossessions,
   buy-to-let portfolio transfers and sales between related parties — including a £569m
   bulk transfer. They are unlearnable noise, and they were destroying the GBP-scale score.
2. **Location needs target encoding, not one-hot.** One-hot produced ~989 sparse columns,
   trained slowly enough to crash the kernel on RandomForest, and scored ~0.18 R² lower.
   Target encoding gives each of six nested geographic levels (`county → area → town_city →
   district → area_code → sector`) one dense, ordered column.

A hyperparameter sweep and street-level encoding were both tried and rejected: each gained
under 0.01 R² while doubling the overfit gap.

### Model comparison

| Model | R² (log) | MdAPE | Overfit gap | Fit time |
|---|---|---|---|---|
| **HistGradientBoosting** | **0.700** | **18.6%** | 0.039 | 8s |
| RandomForest | 0.693 | 18.9% | 0.068 | 35s |
| Ridge | 0.640 | 20.8% | 0.031 | <1s |

### Validation

5-fold cross-validation: `[0.701, 0.696, 0.688, 0.684, 0.697]` → **0.693 ± 0.006**.
The hold-out score sits within one standard deviation of the CV mean, so it is not a lucky
split. Log-scale residuals have mean +0.003 — the model is unbiased.

Error is not uniform. It is lowest mid-market (14–16% in deciles 4–8) and worst at both
extremes (38% in the cheapest decile, 33% in the most expensive), because unusual prices
are exactly where the missing property attributes matter most.

---

## Limitations

**The ceiling is the data, not the model.** Price Paid Data records *what sold for how
much*, not *what was sold*. There is no floor area, bedroom count, garden, or condition in
the file. Two houses on the same street with the same postcode and type are identical to
this model yet can legitimately differ in price by 50%.

Getting meaningfully past R² ≈ 0.70 requires joining in property attributes — the
[EPC register](https://epc.opendatacommunities.org/) provides floor area and habitable room
counts keyed by address — rather than further tuning.

Also note that ~96% of the current file falls in 2025–2026, so `year` carries almost no
trend signal here. On a full historical extract it would matter considerably more.