// server/api/whoami.get.ts   – Nuxt 3 / Nitro endpoint
import { defineEventHandler, getHeader } from 'h3'
import { useRuntimeConfig }              from '#imports'

/**
 * Returns `{ email, username }`
 * •  In prod the e-mail comes from GCP IAP’s `x-goog-authenticated-user-email`
 * •  When developing locally it falls back to runtimeConfig.public.devUserEmail
 */
export default defineEventHandler(event => {
  const cfg        = useRuntimeConfig(event)
  const iapHeader  = getHeader(event, 'x-goog-authenticated-user-email') as string|undefined
  /* ─── ① real IAP header ───────────────────────────────────────── */
  let email = iapHeader ? iapHeader.split(':').pop() : undefined         // “accounts…:alice@foo.io” ⇒ “alice@foo.io”
  /* ─── ② dev fallback (npm run dev) ─────────────────────────────── */
  if (!email && process.env.NODE_ENV === 'development' && cfg.public.devUserEmail) {
    email = cfg.public.devUserEmail            // e.g. set in .env.local  DEV_USER_EMAIL=dev@example.com
  }

  const username = email?.split('@')[0] ?? null
  return { email, username }                    // { email:"alice@foo.io", username:"alice" }
})
