// store/alerts.js – now uses $fetch everywhere (no useFetch after mount)
import { defineStore } from 'pinia'

// import { $fetch } from '#app' // gives TS types & auto-completion

export const useAlertsStore = defineStore('alerts', {
  state: () => ({
    /* dropdown data currently displayed ---------------------------- */
    statesList     : [],
    placesList     : [],
    committeesList : [],

    /* lightweight, per‑session caches ------------------------------ */
    placesCache    : {},   // { "CA":                 ["Los Angeles", …] }
    committeesCache: {},   // { "CA::Los Angeles":     ["City Council", …] }

    /* misc ui‑helper state ----------------------------------------- */
    message: '',
    error  : ''
  }),

  /* ----------------------------------------------------------------
   *  ACTIONS (all network traffic now via $fetch)
   * ---------------------------------------------------------------- */
  actions: {
    /* ==============================================================
     *  CLIENT‑SIDE helpers (used by subscribe.vue)
     * ============================================================== */
    async fetchStates () {
      try {
        this.error = ''
        const { states = [] } = await $fetch('/api/states')
        this.statesList = states
      } catch (err) {
        this.error = `Failed to load states: ${err.message}`
      }
    },

    async fetchPlaces (selectedState) {
      try {
        this.error          = ''
        this.placesList     = []
        this.committeesList = []
        if (!selectedState) return

        /* cache check */
        if (this.placesCache[selectedState]) {
          this.placesList = this.placesCache[selectedState]
          return
        }

        const { places = [] } = await $fetch('/api/places', {
          query: { state: selectedState }
        })
        this.placesCache[selectedState] = places
        this.placesList                 = places
      } catch (err) {
        this.error = `Failed to load places: ${err.message}`
      }
    },

    async fetchCommittees (selectedState, selectedPlace) {
      try {
        this.error          = ''
        this.committeesList = []
        if (!selectedState || !selectedPlace) return

        const key = `${selectedState}::${selectedPlace}`

        /* cache check */
        if (this.committeesCache[key]) {
          this.committeesList = this.committeesCache[key]
          return
        }

        const { committees = [] } = await $fetch('/api/committees', {
          query: { state: selectedState, place: selectedPlace }
        })
        this.committeesCache[key] = committees
        this.committeesList       = committees
      } catch (err) {
        this.error = `Failed to load committees: ${err.message}`
      }
    },

    /* ==============================================================
     *  SERVER‑SIDE helpers (used by meetings.vue)
     *  – same API, just called during SSR then hydrated via Pinia
     * ============================================================== */
    async fetchStatesServerSide () {
      await this.fetchStates() // same logic works SSR & CSR
    },

    async fetchPlacesServerSide (statesIn) {
      try {
        this.error          = ''
        this.placesList     = []
        this.committeesList = []

        const states = Array.isArray(statesIn) ? statesIn : [statesIn]
        if (!states.length) return

        const out = [] // [{ value, state }]
        for (const st of states) {
          if (!st) continue

          if (!this.placesCache[st]) {
            const { places = [] } = await $fetch('/api/places', {
              query: { state: st }
            })
            this.placesCache[st] = places
          }
          this.placesCache[st].forEach(p => out.push({ value:p, state:st }))
        }

        /* dedupe by place value; keep first state tag */
        const seen = new Set()
        this.placesList = out
          .filter(o => !seen.has(o.value) && seen.add(o.value))
          .sort((a,b) => a.value.localeCompare(b.value))
      } catch (err) {
        this.error = `Failed to load places (SSR): ${err.message}`
      }
    },

    async fetchCommitteesServerSide (statesIn, placesIn) {
      try {
        this.error          = ''
        this.committeesList = []

        const states = Array.isArray(statesIn) ? statesIn : [statesIn]
        const places = Array.isArray(placesIn) ? placesIn : [placesIn]
        if (!states.length || !places.length) return

        const out = [] // [{ value, place }]
        for (const st of states) {
          for (const pl of places) {
            if (!st || !pl) continue
            const key = `${st}::${pl}`

            if (!this.committeesCache[key]) {
              const { committees = [] } = await $fetch('/api/committees', {
                query: { state: st, place: pl }
              })
              this.committeesCache[key] = committees
            }
            this.committeesCache[key].forEach(c =>
              out.push({ value:c, place:pl })
            )
          }
        }

        /* dedupe by value + place combo */
        const seen = new Set()
        this.committeesList = out
          .filter(o => {
            const k = `${o.value}::${o.place}`
            return !seen.has(k) && seen.add(k)
          })
          .sort((a,b) => a.value.localeCompare(b.value))
      } catch (err) {
        this.error = `Failed to load committees (SSR): ${err.message}`
      }
    },

    /* ==============================================================
     *  Subscription – POST via $fetch
     * ============================================================== */
    async subscribe (email, selectedState, selectedPlace, selectedCommittees) {
      try {
        this.error   = ''
        this.message = ''

        if (!email || !selectedState || !selectedPlace || !selectedCommittees.length) {
          throw new Error('Please fill out all required fields.')
        }

        const payload = {
          email,
          state     : selectedState,
          place     : selectedPlace,
          committees: selectedCommittees
        }

        await $fetch('/api/subscribe', { method:'POST', body: payload })
        this.message = 'You have been subscribed successfully!'
      } catch (err) {
        this.error = `Failed to subscribe: ${err.message}`
      }
    }
  }
})
