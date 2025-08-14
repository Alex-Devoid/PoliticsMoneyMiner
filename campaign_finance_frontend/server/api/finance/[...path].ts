import { defineEventHandler, getQuery, H3Event } from 'h3'
import { $fetch } from 'ofetch'

/**
 * Build the full URL to the FastAPI service.
 * `financeApiUrl` already has a sensible default in nuxt.config.ts
 *   – `http://backend-campaign-finance-api:8080` for local docker-dev.
 * In prod you’ll set FINANCE_API_URL="https://finance-api.acme.com", etc.
 */
function backendRequest (event: H3Event) {
  const cfg  = useRuntimeConfig()
  const base = cfg.public.financeApiUrl!.replace(/\/$/, '')  // never empty
  const rest = event.context.params!.path                    // sub-path after /finance/
  const query = getQuery(event)

  return { url: `${base}/finance/${rest}`, query }
}

export default defineEventHandler(async (event) => {
  const { url, query } = backendRequest(event)

  /* GET only – all front-end calls are reads */
  return await $fetch(url, { query })
})
