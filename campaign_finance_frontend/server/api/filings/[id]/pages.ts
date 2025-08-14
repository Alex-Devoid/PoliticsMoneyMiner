// server/api/filings/[id]/pages.ts  – drop‑in replacement
import { defineEventHandler, createError,
         getQuery, getRouterParam } from 'h3'
import { useRuntimeConfig } from '#imports'
import { GoogleAuth } from 'google-auth-library'
import fs from 'fs/promises'

export default defineEventHandler(async (event) => {
  const slug   = getRouterParam(event, 'id')!
  const page   = Number(getQuery(event).page || 1)

  /* ── MOCK toggle (unchanged) ── */
  if (process.env.MOCK_DATA?.toLowerCase() === 'on') {
    const json  = await fs.readFile(`mock/${slug}-pages.json`, 'utf8')
    const pages = JSON.parse(json)
    return pages.find(p => p.page === page) || pages[0]
  }

  /* ── live backend ── */
  try {
    const cfg     = useRuntimeConfig(event)
    const baseURL = cfg.public.financeApiUrl!.replace(/\/$/, '')
    const url     = `${baseURL}/finance/filings/${slug}/pages`

    const isDev   = process.env.NODE_ENV === 'development'
    let   token   = ''
    if (!isDev) {
      const auth   = new GoogleAuth()
      const client = await auth.getIdTokenClient(baseURL)
      token        = await client.idTokenProvider.fetchIdToken(baseURL)
    }

    return await $fetch(url, {
      method : 'GET',              // ---- force GET ----
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      },
      query  : { page }            // ---- query‑string only; no body ----
    })
  } catch (err: any) {
    console.error('[filings‑page] FastAPI error:', err)
    throw createError({ statusCode: 500, statusMessage: err.message })
  }
})
