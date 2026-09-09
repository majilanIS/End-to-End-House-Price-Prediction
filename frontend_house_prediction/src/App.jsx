import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import { getHealth, getLocation, getMetrics, getSchema, predict } from './api'

const now = new Date()

const EMPTY_FORM = {
  postcode: '',
  property_type: 'D',
  old_new: 'N',
  duration: 'F',
  year: now.getFullYear(),
  month: now.getMonth() + 1,
}

// Used if /schema cannot be reached, so the form still renders sensible labels.
const FALLBACK_SCHEMA = {
  property_type: {
    D: 'Detached',
    S: 'Semi-detached',
    T: 'Terraced',
    F: 'Flat / Maisonette',
    O: 'Other',
  },
  old_new: { Y: 'Newly built', N: 'Established building' },
  duration: { F: 'Freehold', L: 'Leasehold' },
}

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

const gbp = (value) =>
  new Intl.NumberFormat('en-GB', {
    style: 'currency',
    currency: 'GBP',
    maximumFractionDigits: 0,
  }).format(value)

// Keeps the <html data-theme> attribute, localStorage and the browser-chrome
// colour in sync. The inline script in index.html has already picked the theme
// before React mounts, so this simply owns it from here on.
function useTheme() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme)
  const toggle = useCallback(() => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    document.documentElement.dataset.theme = next
    localStorage.setItem('theme', next)
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', next === 'dark' ? '#0d1117' : '#f5f7fa')
  }, [theme])
  return { theme, toggle }
}

const SunIcon = (
  <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">
    <path
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      d="M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0-15v2m0 16v2M4.2 4.2l1.4 1.4m12.8 12.8 1.4 1.4M2 12h2m16 0h2M4.2 19.8l1.4-1.4m12.8-12.8 1.4-1.4"
    />
  </svg>
)

const MoonIcon = (
  <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">
    <path
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"
    />
  </svg>
)

// A postcode is complete enough to look up once it has an outward code and at
// least the start of the inward code, e.g. "NW10 0".
const LOOKS_COMPLETE = /^[A-Z]{1,2}\d[A-Z\d]?\s*\d/i

function App() {
  const { theme, toggle } = useTheme()
  const [form, setForm] = useState(EMPTY_FORM)
  const [schema, setSchema] = useState(FALLBACK_SCHEMA)
  const [health, setHealth] = useState(null)
  const [metrics, setMetrics] = useState(null)

  const [location, setLocation] = useState(null)
  const [locationState, setLocationState] = useState('idle') // idle | loading | found | unknown
  const [manualLocation, setManualLocation] = useState({ town_city: '', district: '', county: '' })

  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const lookupTimer = useRef(null)

  useEffect(() => {
    getSchema().then(setSchema).catch(() => {})
    getMetrics().then(setMetrics).catch(() => {})
    getHealth()
      .then(setHealth)
      .catch((err) => setHealth({ status: 'unreachable', model_loaded: false, detail: err.message }))
  }, [])

  // ---- Postcode -> location -------------------------------------------------
  // Debounced so typing a postcode does not fire a request per keystroke.
  const lookupPostcode = useCallback((postcode) => {
    clearTimeout(lookupTimer.current)

    if (!LOOKS_COMPLETE.test(postcode)) {
      setLocation(null)
      setLocationState('idle')
      return
    }

    setLocationState('loading')
    lookupTimer.current = setTimeout(async () => {
      try {
        const found = await getLocation(postcode)
        setLocation(found)
        setLocationState(found ? 'found' : 'unknown')
      } catch {
        setLocation(null)
        setLocationState('unknown')
      }
    }, 350)
  }, [])

  useEffect(() => () => clearTimeout(lookupTimer.current), [])

  function handleChange(event) {
    const { name, value } = event.target
    const isNumeric = name === 'year' || name === 'month'
    const next = isNumeric ? Number(value) : value.toUpperCase()

    setForm((previous) => ({ ...previous, [name]: next }))
    if (name === 'postcode') lookupPostcode(next)
  }

  function handleManualChange(event) {
    const { name, value } = event.target
    setManualLocation((previous) => ({ ...previous, [name]: value.toUpperCase() }))
  }

  function optionsFor(field) {
    return Object.entries(schema[field] ?? FALLBACK_SCHEMA[field]).map(([code, label]) => (
      <option key={code} value={code}>
        {label}
      </option>
    ))
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError(null)
    setResult(null)
    setLoading(true)

    // Send location only when the postcode could not resolve it. Otherwise let
    // the backend derive it, which guarantees the values match what the model
    // was trained on.
    const body = { ...form }
    if (locationState === 'unknown') {
      Object.entries(manualLocation).forEach(([key, value]) => {
        if (value.trim()) body[key] = value.trim()
      })
    }

    try {
      setResult(await predict(body))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const apiDown = health && !health.model_loaded
  const typicalError = metrics?.holdout?.MdAPE_pct

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>House Price Predictor</h1>
          <p className="subtitle">UK sale price estimates from HM Land Registry data</p>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={toggle}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? SunIcon : MoonIcon}
        </button>
        {health && (
          <span className={`badge ${health.model_loaded ? 'ok' : 'warn'}`}>
            <span className="dot" />
            {health.model_loaded ? health.model_name : health.status}
          </span>
        )}
      </header>

      {apiDown && (
        <div className="banner error-banner">
          <strong>The API is not serving predictions.</strong>{' '}
          {health.detail ?? 'Start it with: cd backend_house_prediction && python -m app.main'}
        </div>
      )}

      <main className="layout">
        <form className="card form" onSubmit={handleSubmit}>
          <h2 className="card-title">Property details</h2>

          <label className="field">
            <span className="label-row">
              Postcode
              {locationState === 'loading' && <em className="hint">looking up…</em>}
            </span>
            <input
              name="postcode"
              value={form.postcode}
              onChange={handleChange}
              placeholder="NW10 0DY"
              autoComplete="postal-code"
              spellCheck="false"
              required
            />
          </label>

          {locationState === 'found' && location && (
            <div className="resolved">
              <div className="resolved-place">
                {location.district}, {location.county}
              </div>
              <div className="resolved-meta">
                Matched <code>{location.outward_code}</code> · {location.n_sales} sales on record
                {location.confidence < 0.7 && (
                  <span className="ambiguous"> · area spans several districts</span>
                )}
              </div>
            </div>
          )}

          {locationState === 'unknown' && (
            <div className="resolved unknown">
              <div className="resolved-place">Postcode area not in the training data</div>
              <div className="resolved-meta">Enter the location manually for a usable estimate.</div>
              <div className="row manual">
                {['town_city', 'district', 'county'].map((field) => (
                  <label className="field" key={field}>
                    {field === 'town_city' ? 'Town / City' : field === 'district' ? 'District' : 'County'}
                    <input
                      name={field}
                      value={manualLocation[field]}
                      onChange={handleManualChange}
                      placeholder={
                        field === 'town_city' ? 'LONDON' : field === 'district' ? 'BRENT' : 'GREATER LONDON'
                      }
                    />
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="row">
            <label className="field">
              Property type
              <select name="property_type" value={form.property_type} onChange={handleChange}>
                {optionsFor('property_type')}
              </select>
            </label>
            <label className="field">
              Age
              <select name="old_new" value={form.old_new} onChange={handleChange}>
                {optionsFor('old_new')}
              </select>
            </label>
            <label className="field">
              Tenure
              <select name="duration" value={form.duration} onChange={handleChange}>
                {optionsFor('duration')}
              </select>
            </label>
          </div>

          <div className="row">
            <label className="field">
              Month of sale
              <select name="month" value={form.month} onChange={handleChange}>
                {MONTHS.map((label, index) => (
                  <option key={label} value={index + 1}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Year of sale
              <input
                type="number"
                name="year"
                min="1995"
                max="2030"
                value={form.year}
                onChange={handleChange}
                required
              />
            </label>
          </div>

          <button type="submit" disabled={loading || apiDown}>
            {loading ? 'Predicting…' : 'Estimate price'}
          </button>

          {error && <p className="error">{error}</p>}
        </form>

        <section className="card result">
          <h2 className="card-title">Estimate</h2>

          {result ? (
            <>
              <p className="price">{gbp(result.predicted_price)}</p>

              <div className="range">
                <div className="range-bar">
                  <span className="range-fill" />
                </div>
                <div className="range-labels">
                  <span>{gbp(result.lower_bound)}</span>
                  <span>{gbp(result.upper_bound)}</span>
                </div>
                <p className="range-note">Likely range (about two thirds of sales fall here)</p>
              </div>

              <dl className="details">
                <div>
                  <dt>Location used</dt>
                  <dd>
                    {result.district}, {result.county}
                    <span className="source">
                      {result.location_source === 'postcode_lookup'
                        ? ' · from postcode'
                        : result.location_source === 'provided'
                          ? ' · entered manually'
                          : ' · unknown area'}
                    </span>
                  </dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{result.model_name}</dd>
                </div>
              </dl>

              {typicalError && (
                <p className="accuracy">
                  Typical error is about <strong>{typicalError.toFixed(0)}%</strong>. This model knows
                  location, type and tenure — but not floor area, bedrooms or condition, so treat it as
                  a starting point rather than a valuation.
                </p>
              )}
            </>
          ) : (
            <div className="placeholder">
              <p>Enter a postcode and property details to see an estimate.</p>
              {metrics && (
                <ul className="stats">
                  <li>
                    <strong>{(metrics.cross_validation?.R2_mean ?? 0).toFixed(3)}</strong>
                    <span>R² (cross-validated)</span>
                  </li>
                  <li>
                    <strong>{(metrics.holdout?.MdAPE_pct ?? 0).toFixed(1)}%</strong>
                    <span>median error</span>
                  </li>
                  <li>
                    <strong>{(health?.n_training_rows ?? 0).toLocaleString()}</strong>
                    <span>sales trained on</span>
                  </li>
                </ul>
              )}
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        Data: HM Land Registry Price Paid Data · full-market-value sales only · Contains HM Land
        Registry data © Crown copyright and database right.
      </footer>
    </div>
  )
}

export default App
