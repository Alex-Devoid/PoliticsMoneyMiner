// server/api/filings.ts
// --------------------------------------------------------------
// GET /api/filings             → list of candidate filings
// --------------------------------------------------------------

import { defineEventHandler, createError, getQuery } from 'h3'
import { useRuntimeConfig } from '#imports'
import fs from 'fs/promises'
import { GoogleAuth } from 'google-auth-library'

export default defineEventHandler(async (event) => {
  /* ── 0. Mock mode ─────────────────────────────────────────── */
  if (process.env.MOCK_DATA?.toLowerCase() === 'on') {
    try {
      const json = await fs.readFile('mock/filings-overview.json', 'utf8')
      return JSON.parse(json)                       // ← an ARRAY
    } catch {
      throw createError({ statusCode: 500, statusMessage: 'Mock overview missing' })
    }
  }

  /* ── 1. Real backend proxy ────────────────────────────────── */
  try {
    const cfg     = useRuntimeConfig(event)
    const baseURL = cfg.public.financeApiUrl!.replace(/\/$/, '')
    const url     = `${baseURL}/finance/filings`
    const isDev   = process.env.NODE_ENV === 'development'

    /* – optional IAP / Cloud‑Run Auth – */
    let token = ''
    if (!isDev) {
      const auth   = new GoogleAuth()
      const client = await auth.getIdTokenClient(baseURL)
      token        = await client.idTokenProvider.fetchIdToken(baseURL)
    }

    const headers = {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    }

    return await $fetch(url, { headers, query: getQuery(event) })
  } catch (err: any) {
    console.error('[filings] FastAPI error:', err)
    throw createError({ statusCode: 500, statusMessage: err.message })
  }
})
