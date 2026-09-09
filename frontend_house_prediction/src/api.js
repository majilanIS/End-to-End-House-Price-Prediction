// Thin client for the FastAPI backend.
//
// When VITE_API_BASE_URL is empty we use relative paths, which the Vite dev
// server proxies to the backend (see vite.config.js). When it is set, requests
// go straight to that origin — and the backend's ALLOWED_ORIGINS must allow us.
const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const url = (path) => `${BASE_URL}${path}`

/**
 * Turn any error shape the backend can return into one readable string.
 *
 * FastAPI returns `detail` as a plain string for our own HTTPExceptions, but as
 * an ARRAY of objects for Pydantic validation failures (422). Rendering that
 * array directly would show "[object Object]", so unpack it into field messages.
 */
function readError(payload, status) {
  const detail = payload?.detail

  if (typeof detail === 'string') return detail

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = Array.isArray(item.loc)
          ? item.loc.filter((part) => part !== 'body').join('.')
          : null
        return field ? `${field}: ${item.msg}` : item.msg
      })
      .join(' · ')
  }

  return `Request failed (${status})`
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(url(path), options)
  } catch {
    // Network-level failure: the backend is not running, or is unreachable.
    throw new Error(
      'Cannot reach the API. Is the backend running? ' +
        '(cd backend_house_prediction && python -m app.main)',
    )
  }

  // A 204 or an empty body would break .json(), so guard it.
  const text = await response.text()
  const payload = text ? JSON.parse(text) : null

  if (!response.ok) {
    const error = new Error(readError(payload, response.status))
    error.status = response.status
    throw error
  }
  return payload
}

export const getHealth = () => request('/health')
export const getSchema = () => request('/schema')
export const getMetrics = () => request('/metrics')

/** Resolve a postcode to town / district / county. Returns null when unknown. */
export async function getLocation(postcode) {
  try {
    return await request(`/location/${encodeURIComponent(postcode)}`)
  } catch (error) {
    if (error.status === 404) return null // no sales on record — not a failure
    throw error
  }
}

export const predict = (body) =>
  request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
