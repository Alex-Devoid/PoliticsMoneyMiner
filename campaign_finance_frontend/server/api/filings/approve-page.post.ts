// server/api/filings/approve-page.post.ts
import { defineEventHandler, readBody, createError } from 'h3'
import { useRuntimeConfig } from '#imports'
import { GoogleAuth } from 'google-auth-library'

type Row = {
  id: string | number
  field?: string
  label?: string | null
  value?: string | null
  columns?: Record<string, any> | null
  approved?: boolean
}

export default defineEventHandler(async (event) => {
  const body = await readBody<{
    candidateId?: string | number
    slug?: string | number
    id?: string | number
    page?: number | string
    rows?: Row[]
    approved?: boolean
  }>(event)

  const slugRaw = body?.slug ?? body?.candidateId ?? body?.id
  const pageRaw = body?.page
  const rows    = body?.rows

  if (!slugRaw || !pageRaw || !Array.isArray(rows)) {
    throw createError({
      statusCode: 400,
      statusMessage: 'candidateId/slug, page, and rows[] are required'
    })
  }

  const slug = String(slugRaw)
  const page = Number(pageRaw)

  try {
    const cfg     = useRuntimeConfig(event)
    const baseURL = (cfg.public.financeApiUrl || '').replace(/\/$/, '')
    if (!baseURL) throw new Error('Missing runtimeConfig.public.financeApiUrl')

    // auth behavior mirrors server/api/filings/[id]/pages.ts
    const isDev = process.env.NODE_ENV === 'development'
    let token = ''
    if (!isDev) {
      const auth   = new GoogleAuth()
      const client = await auth.getIdTokenClient(baseURL)
      token        = await client.idTokenProvider.fetchIdToken(baseURL)
    }

    // Assumes your FastAPI implements this page-approve endpoint and
    // understands global page + rows edits in one shot.
    const url = `${baseURL}/finance/filings/${encodeURIComponent(slug)}/pages/approve`

    return await $fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      },
      query: { page },
      body : {
        page,
        approved: !!body?.approved,
        rows
      }
    })
  } catch (err: any) {
    console.error('[approve-page] Finance API error:', err)
    throw createError({
      statusCode: err?.statusCode || 500,
      statusMessage: err?.statusMessage || err?.message || 'Approve page failed'
    })
  }
})
