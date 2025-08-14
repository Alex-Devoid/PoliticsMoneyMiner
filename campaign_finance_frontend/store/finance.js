// store/finance.js
import { defineStore } from 'pinia'

/* ───────── helpers ───────── */
const blankSummary     = () => ({
  candidate : { slug: '', name: '', office: '', district: '', jurisdiction: '' },
  cycle     : '',
  metrics   : { raised_total: 0, individual_total: 0, individual_count: 0, pac_total: 0 }
})
const blankSpending    = () => ({ spent_total: 0, txns: 0 })
const blankTopDonors   = () => ({ individual: [], pac: [], cycle: '' })
const blankRaisedSince = () => ({ raised_total: 0, txns: 0, since: '', slug: '' })

/* ───────── store ───────── */
export const useFinanceStore = defineStore('finance', {
  state: () => ({
    /*  search page  */
    candidateHits : [],
    donorHits     : [],

    /*  candidate-detail  */
    summary       : blankSummary(),
    spending      : blankSpending(),
    topDonors     : blankTopDonors(),
    raisedSince   : blankRaisedSince(),

    /*  misc  */
    error         : ''
  }),

  actions: {
    /* generic lookup (search & lists) */
    async _fetchLookup (endpoint, q = '', stateFilter) {
      const query = {}
      if (q && q.trim().length >= 2) query.q = q.trim()
      if (stateFilter)               query.state = stateFilter

      try {
        this.error = ''
        const { hits = [] } = await $fetch(endpoint, { query })
        return hits
      } catch (err) {
        this.error = `Lookup failed: ${err.message}`
        return []
      }
    },

    /* ───────── search-page actions ───────── */
    async searchCandidates (q, state = null) {
      this.candidateHits = await this._fetchLookup(
        '/api/finance/lookup-candidates',
        q,
        state
      )
    },

    async searchDonors (q, state = null) {
      this.donorHits = await this._fetchLookup(
        '/api/finance/lookup-donors',
        q,
        state
      )
    },

    

    /* full list for one state (no "q") */
    async fetchCandidatesForState (stateCode) {
      try {
        this.error = ''
        const { hits = [] } = await $fetch('/api/finance/list-candidates', {
          query: { state: stateCode }
        })
        this.candidateHits = hits
      } catch (err) {
        this.error = `Candidate list failed: ${err.message}`
        this.candidateHits = []
      }
    },

    /* ───────── candidate-detail actions ───────── */
    async fetchCandidateSummary (slug, cycle = undefined) {
      this.summary = blankSummary()
      this.error   = ''
      try {
        const data = await $fetch(
          `/api/finance/candidates/${encodeURIComponent(slug)}/summary`,
          { query: cycle ? { cycle } : {} }
        )
        this.summary = data
        return data
      } catch (err) {
        this.error = `Summary failed: ${err.message}`
        return null
      }
    },

    async fetchCandidateSpending (slug, cycle) {
      this.spending = blankSpending()
      this.error    = ''
      try {
        const data = await $fetch(
          `/api/finance/candidates/${encodeURIComponent(slug)}/spending`,
          { query: cycle ? { cycle } : {} }
        )
        this.spending = data
        return data
      } catch (err) {
        this.error = `Spending failed: ${err.message}`
        return null
      }
    },

    async fetchTopDonors (slug, cycle) {
      this.topDonors = blankTopDonors()
      this.error     = ''
      try {
        const data = await $fetch(
          `/api/finance/candidates/${encodeURIComponent(slug)}/top-donors`,
          { query: cycle ? { cycle } : {} }
        )
        this.topDonors = data
        return data
      } catch (err) {
        this.error = `Top-donors failed: ${err.message}`
        return null
      }
    },

    async fetchRaisedSince (slug, afterISO) {
      this.raisedSince = blankRaisedSince()
      this.error       = ''
      try {
        const data = await $fetch(
          `/api/finance/candidates/${encodeURIComponent(slug)}/raised-since`,
          { query: { after: afterISO } }
        )
        this.raisedSince = data
        return data
      } catch (err) {
        this.error = `Raised-since failed: ${err.message}`
        return null
      }
    },

    /* aliases expected by [slug].vue */
    fetchCandidateTopDonors   (slug, cycle) { return this.fetchTopDonors(slug, cycle) },
    fetchCandidateRaisedSince (slug, after) { return this.fetchRaisedSince(slug, after) }
  }
})
