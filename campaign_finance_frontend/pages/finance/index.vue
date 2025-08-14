<!-- pages/finance/index.vue -->
<template>
  <v-app>
    <NavBar />

    <v-main>
      <v-container class="text-center">
        <h1 class="text-h4 font-weight-medium mb-6">POLITICS MONEY MINER</h1>

        <!-- ── State filter (optional) ───────────────────────────── -->
        <v-row justify="center">
          <v-col cols="12" md="3">
            <v-select
              id="finance-state-select"
              v-model="selectedState"
              :items="statesList"
              :item-title="s => s.toUpperCase()"
              label="Filter by state (optional)"
              clearable
              density="compact"
            />
          </v-col>
        </v-row>

        <v-row justify="center" align="center">
          <!-- Candidate search -->
          <v-col cols="12" md="4">
            <v-autocomplete                                 
              id="finance-candidate-select"
              v-model="selectedCandidate"
              v-model:search="candSearch"
              v-model:menu="candMenuOpen"                   
              :items="candHits"
              :item-title="h => `${h.name} — ${h.office}`"
              :item-value="h => h.slug ?? h.name"
              
              :custom-filter="() => true"                   
              label="Candidate name"
              clearable
              density="compact"
              hide-no-data
              @update:modelValue="goCandidate"
            />
          </v-col>

          <!-- Donor search -->
          <v-col cols="12" md="4">
            <v-text-field
              id="finance-donor-search"
              v-model="donorQuery"
              label="Donor name"
              append-inner-icon="mdi-magnify"
              @keyup.enter="searchDonors"
              clearable
            />
            <v-list v-if="donorHits.length && donorQuery">
              <v-list-item
                v-for="hit in donorHits"
                :key="hit.name"
                @click="goDonor(hit.name)"
              >
                <v-list-item-title>{{ hit.name }}</v-list-item-title>
                <v-list-item-subtitle>
                  {{ hit.is_org ? 'PAC/Org' : 'Individual' }}
                </v-list-item-subtitle>
              </v-list-item>
            </v-list>
          </v-col>
        </v-row>

        <v-alert v-if="store.error" type="error" class="mt-4">
          {{ store.error }}
        </v-alert>
      </v-container>
    </v-main>

    <AppFooter />
  </v-app>
</template>

<script setup>
/* ───────────── imports ───────────── */
import { ref, computed, watch } from 'vue'
import { useAsyncData } from '#app'
import { useRouter } from 'vue-router'

import { useAlertsStore } from '@/store/alerts'
import { useFinanceStore } from '@/store/finance'

/* ─────────── debounce util ───────── */
function debounce (fn, delay = 300) {
  let h
  return (...args) => {
    clearTimeout(h)
    h = setTimeout(() => fn(...args), delay)
  }
}

/* ─────────── stores & router ─────── */
const router = useRouter()
const alerts = useAlertsStore()
const store  = useFinanceStore()

/* ─────────── local refs ──────────── */
const selectedState     = ref(null)
const selectedCandidate = ref(null)      // slug
const candSearch        = ref('')        // text inside the autocomplete
const candMenuOpen      = ref(false)     // menu visibility
const donorQuery        = ref('')        // donor search box

/* ─────────── preload states ──────── */
const { data: statesRaw } = await useAsyncData('finance-states',
  () => $fetch('/api/finance/states')
)
alerts.statesList = Array.isArray(statesRaw.value)
  ? statesRaw.value
  : statesRaw.value?.states ?? []
const statesList = computed(() => alerts.statesList)

/* ─────────── debounced lookups ───── */
const debCand  = debounce(txt => store.searchCandidates(txt, selectedState.value))
const debDonor = debounce(txt => store.searchDonors   (txt, selectedState.value))

/* ─────────── state change handler ── */
watch(selectedState, st => {
  selectedCandidate.value = null
  candSearch.value        = ''
  donorQuery.value        = ''
  candMenuOpen.value      = false       // keep menu closed

  if (st) {
    /* full list for the new state (empty q ⇒ wildcard) */
    store.fetchCandidatesForState(st)
  } else {
    store.candidateHits = []
  }
})

/* ─────────── live filtering ──────── */
watch(candSearch, txt => {
  if (txt && txt.length > 1) {
    candMenuOpen.value = true           // show matches while typing
    debCand(txt)
  } else {
    candMenuOpen.value = false
  }
})
watch(donorQuery, txt => { if (txt?.length > 1) debDonor(txt) })

/* ─────────── computed options ────── */
const candHits  = computed(() => store.candidateHits)
const donorHits = computed(() => store.donorHits)

/* ─────────── manual searches (Enter) */
function searchCandidates () {
  store.searchCandidates(candSearch.value, selectedState.value)
  candMenuOpen.value = true
}
function searchDonors () {
  store.searchDonors(donorQuery.value, selectedState.value)
}

/* ─────────── navigation helpers ──── */
function goCandidate (slug) {
  if (slug) router.push(`/finance/candidate/${slug}`)
}
function goDonor (name) {
  if (name) router.push(`/finance/donor/${encodeURIComponent(name)}`)
}
</script>
