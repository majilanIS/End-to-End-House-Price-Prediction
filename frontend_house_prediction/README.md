# Frontend — House Price Predictor

React + Vite client for the FastAPI backend in `../backend_house_prediction`.

## Run

```bash
npm install
cp .env.example .env      # optional; defaults work as-is
npm run dev               # http://localhost:5173
```

The backend must be running too:

```bash
cd ../backend_house_prediction && python -m app.main
```

| Script | Does |
|---|---|
| `npm run dev` | dev server with hot reload + API proxy |
| `npm run build` | production bundle into `dist/` |
| `npm run preview` | serve the built bundle (port 4173) |
| `npm run lint` | oxlint |

## How it talks to the API

`VITE_API_BASE_URL` decides:

- **Empty (default)** — the app calls relative paths (`/predict`, `/location/...`) and the
  Vite dev server proxies them to `VITE_DEV_API_TARGET`. The browser stays on one origin,
  so CORS never applies. Best for development.
- **Set** — requests go straight to that URL. The backend's `ALLOWED_ORIGINS` must then
  include this app's origin.

Everything goes through `src/api.js`, which also unpacks FastAPI's two different error
shapes: a plain string for application errors, and an array of field objects for Pydantic
validation failures (422). Rendering the latter naively shows `[object Object]`.

## Postcode autofill

Typing a postcode calls `GET /location/{postcode}` (debounced 350 ms) and the backend
returns the town, district and county **from its own training data**. Those fields are
target-encoded, so they must match the training vocabulary exactly — deriving them removes
the possibility of a typo silently producing a confident but wrong estimate.

If the postcode area has no sales on record the app says so and reveals manual location
inputs instead.

## Notes

- Light and dark themes follow the OS setting; colours are CSS custom properties defined
  once in `src/index.css`.
- The estimate is shown with its likely range and the model's typical error, because a bare
  point estimate overstates the precision of a model with ~18.6% median error.
