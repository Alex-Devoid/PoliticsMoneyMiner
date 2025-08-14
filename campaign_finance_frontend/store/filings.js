// store/filings.js
// -----------------------------------------------------------------------------
// Pinia store for the Filings QC dashboard
// – reads the campaign-finance FastAPI (+ Firestore) service via Nuxt server APIs
// – MOCK_DATA=on still works via the Nuxt /api/filings handler
// – PAGE-ONLY approval (no row-level endpoints)
// -----------------------------------------------------------------------------

import { defineStore } from 'pinia'

/*  Centralised 405->POST shim (kept for compatibility)
    The Nuxt route for “/api/filings/:slug/pages” may use GET in dev/mock
    and POST in prod. This shim retries as POST only on 405. */
async function $fetchMaybePOST (url, opts = {}) {
  try {
    return await $fetch(url, opts)
  } catch (err) {
    if (err?.status !== 405) throw err
    return await $fetch(url, { ...opts, method: 'POST' })
  }
}

export const useFilingsStore = defineStore('filings', {
  state: () => ({
    /* ── overview list ───────────────────────────────────────── */
    /** @type {{ id:string, name:string, office:string,
     *            totalPages:number, approved:number }[]} */
    list: [],

    /* ── page-level payload ──────────────────────────────────── */
    currentSlug   : null,
    backendDocId  : null,   // Firestore doc_id for writes (from backend)
    pageImg       : null,   // data-URL PNG or public URL
    rows          : [],     // [{ id,label,value,bbox,approved,columns?,_dirty? }]
    bboxes        : [],     // derived [{ id,x0,y0,x1,y1 }]
    totalPages    : 0,
    currentPageNum: 1,
    currentId     : null,   // active row id in <v-table>
    filesMap      : null,
    approvedPages : {},     // NEW: { [globalPageNumber]: true|false }

    /* ── UX flags ───────────────────────────────────────────── */
    loading       : false,
    error         : '',
  }),

  getters: {
    raceGroups (s) {
      const map = new Map()
      s.list.forEach(c => {
        if (!map.has(c.office)) map.set(c.office, [])
        map.get(c.office).push(c)
      })
      return Array.from(map, ([name, candidates]) => ({ name, candidates }))
    },
    isApproved: s => id => s.rows.find(r => r.id === id)?.approved ?? false,
  },

  actions: {
    /* 1️⃣  overview cards -------------------------------------- */
    async fetchList () {
      this.error = ''
      try {
        this.list = await $fetch('/api/filings')
      } catch (e) {
        console.error('[FilingsStore] fetchList', e)
        this.error = 'Could not load filings'
      }
    },

    /* 2️⃣  one page of one candidate --------------------------- */
    async fetchPage (candidateSlug, page = 1) {
      this.error   = ''
      this.loading = true
      try {
        const res = await $fetchMaybePOST(
          `/api/filings/${candidateSlug}/pages`,
          { query: { page } }
        )

        // ▼ state update ----------------------------------------
        this.currentSlug    = candidateSlug
        this.backendDocId   = res.doc_id
        this.pageImg        = res.pageImg
        this.rows           = (res.rows ?? []).sort(
          (a, b) => (a.row_order ?? 0) - (b.row_order ?? 0)
        ).map(r => ({ ...r, _dirty: false })) // normalize _dirty flag

        this.bboxes = this.rows
          .filter(r => Array.isArray(r.bbox) && r.bbox.length === 4)
          .map(r => {
            const [x0, y0, x1, y1] = r.bbox
            return { id: r.id, x0, y0, x1, y1 }
          })

        this.totalPages     = res.totalPages
        this.filesMap       = res.files
        this.currentPageNum = page
        this.currentId      = this.rows[0]?.id ?? null

        // NEW: per-page approval flags for the sidebar thumbs
        this.approvedPages  = res.approvedPages || {}
      } catch (e) {
        console.error('[FilingsStore] fetchPage', e)
        this.error = 'Unable to load that page'
      } finally {
        this.loading = false
      }
    },

    /* 3️⃣  inline edit helpers (no immediate network writes) --- */
    updateValue (id, val) {
      const r = this.rows.find(r => r.id === id)
      if (r) { r.value = val; r._dirty = true }
    },

    /* If you edit structured columns in the UI, call this */
    updateColumn (id, key, val) {
      const r = this.rows.find(r => r.id === id)
      if (!r) return
      if (!r.columns) r.columns = {}
      r.columns[key] = val
      r._dirty = true
    },

    /* 4️⃣  page-only approval ---------------------------------- */
    approvePage (slugFromCaller) {
      try {
        // 1) Coerce slug & page
        const slug = String(slugFromCaller ?? this.currentSlug ?? '').trim()
        const page = Number(this.currentPageNum || 0)

        // 2) Build patches (always an array, possibly empty)
        const patches = Array.isArray(this.rows)
          ? this.rows
              .filter(r => r && (r._dirty || !r.approved))
              .map(r => {
                const out = { row_id: r.id, approved: true }
                if ('value' in r) out.value = r.value
                if (r.columns && typeof r.columns === 'object') out.columns = r.columns
                return out
              })
          : []

        // 3) Defensive checks to satisfy API contract
        if (!slug) throw new Error('Missing slug')
        if (!Number.isFinite(page) || page < 1) throw new Error('Missing/invalid page')

        // 4) POST to Nuxt API — always include rows as an array (even empty)
        return $fetch('/api/filings/approve-page', {
          method: 'POST',
          body: {
            // send multiple keys for maximum compatibility with the API’s validator
            slug,
            candidateId: slug,
            id: slug,

            page,
            approved: true,
            rows: patches ?? []         // <-- must be an Array
          }
        }).then(() => {
          // optimistic UI
          if (Array.isArray(this.rows)) {
            this.rows.forEach(r => { r.approved = true; r._dirty = false })
          }
          // NEW: mark the current page approved for the sidebar highlight
          this.approvedPages = { ...(this.approvedPages || {}), [page]: true }
        }).catch(e => {
          console.error('[FilingsStore] approvePage', e)
          this.error = 'Could not approve page'
        })
      } catch (e) {
        console.error('[FilingsStore] approvePage (precheck)', e)
        this.error = (e && e.message) || 'Approve page failed'
      }
    },

    /* 5️⃣  misc ------------------------------------------------- */
    thumbSrc (n) {
      const url = this.filesMap?.[`page_${n}`]
      return url?.startsWith('gs://')
        ? url.replace(/^gs:\/\//, 'https://storage.googleapis.com/')
        : url || this.pageImg
    },
    isPageApproved (n) {
      return !!this.approvedPages?.[Number(n)]
    },
    selectNext () {
      const i = this.rows.findIndex(r => r.id === this.currentId)
      if (i < this.rows.length - 1) this.currentId = this.rows[i + 1].id
    },
    selectPrev () {
      const i = this.rows.findIndex(r => r.id === this.currentId)
      if (i > 0) this.currentId = this.rows[i - 1].id
    },
  },
})
