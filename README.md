# House Price Prediction — End to End

Predicts UK residential sale prices from [HM Land Registry Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads),
served through a FastAPI backend with a React frontend.

| | |
|---|---|
| **Model** | HistGradientBoostingRegressor on `log1p(price)` |
| **R² (log scale)** | 0.700 hold-out · **0.693 ± 0.006** 5-fold cross-validated |
| **Median absolute error** | 18.6% |
| **Training rows** | 81,066 full-market-value sales |

---

## Project structure

```
House Price Prediction end-to-end/
│
├── backend_house_prediction/
│   ├── notebooks/
│   │   └── house_price_prediction.ipynb   # EDA, model comparison, validation
│   │
│   ├── src/
│   │   ├── config.py                      # settings from .env
│   │   ├── preprocessing.py               # cleaning + features (shared by training AND API)
│   │   ├── train.py                       # trains, validates, writes the model bundle
│   │   └── predict.py                     # loads the bundle, returns a price
│   │
│   ├── app/
│   │   └── main.py                        # FastAPI
│   │
│   ├── models/
│   │   ├── house_price_model.pkl          # joblib bundle (git-ignored)
│   │   └── metrics.json                   # scores of the saved model
│   │
│   ├── data/                              # raw + cleaned CSVs (git-ignored)
│   ├── .env / .env.example
│   ├── .dockerignore
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend_house_prediction/             # React + Vite
│   ├── src/
│   │   ├── api.js                         # backend client
│   │   ├── App.jsx                        # prediction form + result + theme toggle
│   │   ├── App.css
│   │   └── index.css                      # design tokens, light + dark
│   ├── .env / .env.example
│   ├── .dockerignore
│   ├── Dockerfile                         # builds the SPA, serves it via nginx
│   ├── nginx.conf                         # static files + API reverse proxy
│   ├── package.json
│   └── vite.config.js
│
├── docker-compose.yml                     # one-origin deployment of both halves
├── .env.example                           # vars for docker compose
├── .gitignore
└── README.md
```

`src/preprocessing.py` is deliberately shared: training and serving build features with the
same code, which is what stops the classic "works in the notebook, wrong in production" bug.

---

## Quick start

### Backend

```bash
cd backend_house_prediction

# A project-local virtualenv keeps these dependencies off your global Python
# and makes `python` unambiguous inside this folder.
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)

pip install -r requirements.txt       # runtime: training + API
pip install -r requirements-dev.txt   # optional: notebook, matplotlib, seaborn

# Put the Price Paid CSV in data/ (see "Data" below), then:
python -m src.train        # ~1 min; writes models/house_price_model.pkl
python -m app.main         # serves the API — see /docs
```

`python -m app.main` reads host, port and reload from `.env`.

**If a command is "not recognized" or a module is missing**, the virtualenv is almost
certainly not active — the shell is falling back to a different Python that lacks these
packages. Either activate it, or call it directly without activating:

```powershell
.venv\Scripts\python.exe -m app.main
```

Two things that trip people up:

- It is `app.main` (a *module path*), never `main.py`. `python -m main.py` fails.
- `uvicorn app.main:app --reload` is equivalent, but only works when the venv is active,
  because `uvicorn.exe` lives in `.venv\Scripts`. `python -m uvicorn app.main:app --reload`
  works either way.

### Frontend

```bash
cd frontend_house_prediction
npm install
npm run dev                            # http://localhost:5173
```

The backend already allows `localhost:5173` via CORS. For deployment, set
`ALLOWED_ORIGINS="https://your-domain.com"` rather than widening it to `*`.

The frontend has a light/dark theme toggle in the header. The first visit follows
your OS preference; picking a theme manually saves it in `localStorage`.

### Docker (single service)

```bash
cd backend_house_prediction
python -m src.train                    # the image copies models/ in, it does not train
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

### Docker Compose (full app, one origin — recommended for deployment)

```bash
cp .env.example .env                   # optional; all values have defaults
docker compose up --build -d
# open http://localhost:8080
```

This builds both halves and serves the whole app on **one origin**: nginx serves
the static frontend and reverse-proxies `/health`, `/schema`, `/metrics`,
`/location` and `/predict` to the FastAPI backend. Because the browser only ever
talks to one host, no CORS configuration is involved.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | public port of the app |
| `VITE_API_BASE_URL` | *(empty)* | empty → proxied through nginx; set only if the API lives elsewhere |
| `ALLOWED_ORIGINS` | `http://localhost:8080` | CORS origins (unused while proxied) |

The backend image ships the trained model, so **run `python -m src.train` once
before `docker compose build`** — otherwise the API starts in "degraded" mode
(see `/health`) and the frontend shows a warning banner.

---

## Deploying (Render, Fly, any Docker host)

**The trained model is committed to the repo** (`models/house_price_model.pkl`, ~1.4 MB).
That is deliberate. The training CSV is far too large for git, so a deploy host cannot
retrain — it can only run what we ship. The model and the code that loads it therefore have
to be versioned together.

**scikit-learn is pinned exactly** (`scikit-learn==1.9.0`). A pickled sklearn Pipeline is
not portable across minor versions: a model trained on 1.8 fails to load on 1.9 with the
famously unhelpful

```
ModuleNotFoundError: No module named '_loss'
```

The bundle records the version that built it, and `load_bundle()` compares it against the
installed one at startup, so a mismatch now reports the real cause instead of that message.
The API stays up and marks itself `degraded` on `/health` rather than crash-looping.

**When you change the pinned scikit-learn version you must retrain and commit the model:**

```bash
python -m src.train
git add models/house_price_model.pkl models/metrics.json
```

> **If a deploy fails after the model looked fine locally, clear the host's build cache.**
> A cached Docker layer can keep serving a model file from an older build, which is exactly
> how a 1.8-pickled model ended up running against a 1.9 install. On Render:
> *Manual Deploy → Clear build cache & deploy*.

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
| `GET`  | `/health` | is the model loaded, when was it trained, active config |
| `GET`  | `/schema` | valid category codes, so a client can build a form |
| `GET`  | `/metrics` | hold-out and cross-validation scores of the deployed model |
| `GET`  | `/location/{postcode}` | resolve a postcode to town / district / county |
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
  "predicted_price": 367209.44,
  "lower_bound": 260446.00,
  "upper_bound": 517738.00,
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

## Configuration

Both halves read a `.env` file; both run without one. Copy the examples to start:

```bash
cp backend_house_prediction/.env.example  backend_house_prediction/.env
cp frontend_house_prediction/.env.example frontend_house_prediction/.env
```

**Backend** (`backend_house_prediction/.env`) — real environment variables always win, so
a container's settings are never overridden by a stray file.

| Variable | Default | Purpose |
|---|---|---|
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | where the server binds |
| `RELOAD` | `false` | auto-reload on code changes (development only) |
| `ALLOWED_ORIGINS` | Vite dev + preview | comma-separated CORS origins |
| `MODEL_PATH` | `models/house_price_model.pkl` | model bundle to serve |
| `DATA_PATH` | `data/pp-monthly-update-new-version.csv` | training data |
| `RANDOM_STATE`, `TEST_SIZE`, `CV_FOLDS` | `42`, `0.2`, `5` | training knobs |

**Frontend** (`frontend_house_prediction/.env`) — only `VITE_`-prefixed variables reach the
browser, and they are baked into the built JavaScript, so never put secrets here.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | *(empty)* | Empty → relative paths + the Vite dev proxy, so the browser stays on one origin and CORS never applies. Set it (e.g. `https://api.your-domain.com`) for a deployed build — then add that app's origin to the backend's `ALLOWED_ORIGINS`. |
| `VITE_DEV_API_TARGET` | `http://127.0.0.1:8000` | where the dev proxy forwards |

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
