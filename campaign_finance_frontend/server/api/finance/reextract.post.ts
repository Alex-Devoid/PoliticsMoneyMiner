// server/api/finance/reextract.post.ts
import { defineEventHandler, readBody, createError } from 'h3'
import { useRuntimeConfig } from '#imports'
import { GoogleAuth } from 'google-auth-library'

// Lazy dispatcher cache (no build-time dependency on 'undici')
let _dispatcher: any | null = null
let _dispatcherTried = false
async function getDispatcher () {
  if (_dispatcherTried) return _dispatcher
  _dispatcherTried = true
  try {
    // Only resolved at runtime; safe if 'undici' is not installed
    const { Agent } = await import('undici')
    _dispatcher = new Agent({
      headersTimeout: 300_000, // wait up to 5 min for response headers
      bodyTimeout: 0,          // don't force a body timeout
      connectTimeout: 60_000
    })
  } catch {
    _dispatcher = null
  }
  return _dispatcher
}

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const cfg = useRuntimeConfig(event)
  const baseURL = cfg.public.financeApiUrl!.replace(/\/$/, '')
  const url = `${baseURL}/finance/reextract/rotate-extract`

  const isDev = process.env.NODE_ENV === 'development'
  let token = ''
  if (!isDev) {
    const auth = new GoogleAuth()
    const client = await auth.getIdTokenClient(baseURL)
    token = await client.idTokenProvider.fetchIdToken(baseURL)
  }

  // Prepare a fetch that uses the undici dispatcher when available
  const dispatcher = await getDispatcher()
  const fetchWithOptionalDispatcher = (input: any, init?: any) =>
    globalThis.fetch(input, dispatcher ? { ...init, dispatcher } : init)

  try {
    return await $fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      },
      body,
      // keep the request alive long enough for the backend/OpenAI call
      retry: 0,
      timeout: 0,             // disable ofetch’s abort timer
      responseType: 'json',
      fetch: fetchWithOptionalDispatcher
    })
  } catch (err: any) {
    console.error('[reextract] FastAPI error:', err)
    throw createError({
      statusCode: err?.status || 500,
      statusMessage: err?.data?.detail || err.message
    })
  }
})
