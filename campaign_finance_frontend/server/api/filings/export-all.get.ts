// server/api/filings/export-all.get.ts
import { defineEventHandler } from 'h3'
import { useRuntimeConfig } from '#imports'
import { GoogleAuth } from 'google-auth-library'

export default defineEventHandler(async (event) => {
  const cfg  = useRuntimeConfig(event)
  const baseURL = (cfg.public.financeApiUrl || '').replace(/\/$/, '')
  if (!baseURL) {
    event.node.res.statusCode = 500
    return 'Missing financeApiUrl'
  }

  const isDev = process.env.NODE_ENV === 'development'
  let token = ''
  if (!isDev) {
    const auth   = new GoogleAuth()
    const client = await auth.getIdTokenClient(baseURL)
    token        = await client.idTokenProvider.fetchIdToken(baseURL)
  }

  const url = `${baseURL}/finance/filings/export-all?format=csv`
  const csv = await $fetch<string>(url, {
    method: 'GET',
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    }
  })

  event.node.res.setHeader('Content-Type', 'text/csv; charset=utf-8')
  event.node.res.setHeader('Content-Disposition', 'attachment; filename="all_candidates.csv"')
  return csv
})
