// server/api/_utils.ts
import { GoogleAuth } from 'google-auth-library'
import { useRuntimeConfig } from '#imports'
import { createError, getQuery } from 'h3'

// choose the right base URL ------------------------------------------------
function pickBaseUrl (path: string, cfg: any) {
  return path.startsWith('/finance/')
    ? cfg.public.financeApiUrl          // campaign-finance container
    : cfg.public.apiUrl                 // meetings / alerts container
}

export async function callFastApi (
  event,
  path: string,
  query: Record<string, any> = {},
  opts: { method?: 'GET'|'POST', body?: any } = {}
) {
  const config   = useRuntimeConfig(event)
  const baseUrl  = pickBaseUrl(path, config)
  const fastUrl  = `${baseUrl}${path}`
  const isDev    = process.env.NODE_ENV === 'development'
  let   idToken  = ''

  /* -- IAP token only in prod ----------------------------------- */
  if (!isDev) {
    const auth   = new GoogleAuth()
    const client = await auth.getIdTokenClient(baseUrl)
    idToken      = await client.idTokenProvider.fetchIdToken(baseUrl)
  }

  const headers = {
    'Content-Type': 'application/json',
    ...(idToken ? { Authorization: `Bearer ${idToken}` } : {})
  }

  try {
    return await $fetch(fastUrl, {
      method : opts.method ?? 'GET',
      headers,
      query ,
      body   : opts.body
    })
  } catch (err: any) {
    console.error('FastAPI proxy error:', err)
    throw createError({
      statusCode   : 500,
      statusMessage: `FastAPI Error: ${err.message}`
    })
  }
}
