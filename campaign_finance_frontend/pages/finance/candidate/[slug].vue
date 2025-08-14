<!-- pages/finance/candidate/[slug].vue -->
<template>
  <v-app>
    <NavBar />

    <v-main>
      <!-- ───────── LOADED STATE ───────── -->
      <v-container v-if="summary">
        <h2 class="text-h3 font-weight-bold mb-6">
          {{ summary.candidate.name }}
        </h2>

        <v-row>
          <!--  ❙  Cycle metrics  -->
          <v-col cols="12" md="6">
            <v-card outlined>
              <v-card-title class="text-h5">
                This cycle ({{ summary.cycle }})
              </v-card-title>

              <v-card-text v-if="metrics">
                <div>
                  Amount&nbsp;raised:
                  <strong>${{ fmt(metrics.raised_total) }}</strong>
                </div>
                <div>From individuals: ${{ fmt(metrics.individual_total) }}</div>
                <div>From PACs: ${{ fmt(metrics.pac_total) }}</div>
                <div>
                  Avg. individual gift:
                  {{ metrics.individual_count ? `$${fmt(avgInd)}` : '—' }}
                </div>

                <div class="mt-4">
                  Spending: ${{ fmt(spending?.spent_total ?? 0) }}
                </div>
              </v-card-text>

              <v-card-text v-else>
                <v-progress-circular indeterminate />
              </v-card-text>
            </v-card>
          </v-col>

          <!--  ❙  Top donors  -->
          <v-col cols="12" md="6">
            <v-card outlined>
              <v-card-title class="text-h5">Top donors</v-card-title>

              <v-card-text v-if="topDonors">
                <h4 class="text-subtitle-1 mt-0">PAC / Orgs</h4>
                <ul>
                  <li v-for="d in topDonors.pac.slice(0, 5)" :key="d.name">
                    {{ d.name }} — ${{ fmt(d.total) }}
                  </li>
                </ul>

                <h4 class="text-subtitle-1">Individuals</h4>
                <ul>
                  <li v-for="d in topDonors.individual.slice(0, 5)" :key="d.name">
                    {{ d.name }} — ${{ fmt(d.total) }}
                  </li>
                </ul>
              </v-card-text>

              <v-card-text v-else>
                <v-progress-circular indeterminate />
              </v-card-text>
            </v-card>
          </v-col>
        </v-row>

        <!--  ❙  Raised-since demo  -->
        <v-row class="mt-8">
          <v-col cols="12" md="4">
            <v-text-field
              type="date"
              v-model="sinceDate"
              label="Raised since…"
            />
          </v-col>

          <v-col cols="12" md="2">
            <v-btn color="primary" @click="refetchSince">Go</v-btn>
          </v-col>

          <v-col cols="12" md="6" v-if="sinceTotal">
            <p class="text-body-1">
              Raised since {{ sinceDate }}:
              <strong>${{ fmt(sinceTotal.raised_total) }}</strong>
              ({{ sinceTotal.txns }} contributions)
            </p>
          </v-col>
        </v-row>

        <v-alert v-if="store.error" type="error" class="mt-4">
          {{ store.error }}
        </v-alert>
      </v-container>

      <!-- ───────── LOADING STATE ───────── -->
      <v-container v-else>
        <v-progress-circular indeterminate />
      </v-container>
    </v-main>

    <AppFooter />
  </v-app>
</template>

<script setup>
/* ───── imports ───── */
import { ref, computed, watch } from 'vue'
import { useRoute }             from 'vue-router'

import NavBar     from '@/components/NavBar.vue'
import AppFooter  from '@/components/AppFooter.vue'
import { useFinanceStore } from '@/store/finance'

/* ───── route & store ───── */
const route   = useRoute()
const store   = useFinanceStore()
const slugRef = ref(route.params.slug)   // keep reactive for route changes

/* ───── local refs ───── */
const summary     = ref(null)
const spending    = ref(null)
const topDonors   = ref(null)
const sinceDate   = ref('')
const sinceTotal  = ref(null)

/* ───── helper: money format ───── */
const fmt = (v) => Number(v || 0).toLocaleString()

/* ───── derived helpers ───── */
const metrics = computed(() => summary.value?.metrics ?? null)
const avgInd  = computed(() =>
  metrics.value && metrics.value.individual_total && metrics.value.individual_count
    ? (metrics.value.individual_total / metrics.value.individual_count).toFixed(2)
    : 0
)

/* ───── core fetch chain ───── */
async function loadCandidate (slug) {
  // reset UI
  summary.value  = null
  spending.value = null
  topDonors.value= null
  sinceTotal.value = null
  store.error = ''

  /* 1️⃣  Summary first (no cycle param) */
  const sum = await store.fetchCandidateSummary(slug)
  if (!sum) return                       // bubbled error handled by store

  summary.value = sum

  /* 2️⃣  Use the cycle the API decided on */
  const cyc = sum.cycle

  // parallel fetches
  const [sp, td] = await Promise.all([
    store.fetchCandidateSpending (slug, cyc),
    store.fetchTopDonors         (slug, cyc)
  ])
  spending.value  = sp
  topDonors.value = td
}

/* ───── raised-since helper ───── */
async function refetchSince () {
  if (!sinceDate.value) return
  sinceTotal.value = null
  const data = await store.fetchRaisedSince(slugRef.value, sinceDate.value)
  sinceTotal.value = data
}

/* ───── initial & reactive load ───── */
await loadCandidate(slugRef.value)
watch(() => route.params.slug, (newSlug) => {
  if (newSlug && newSlug !== slugRef.value) {
    slugRef.value = newSlug
    loadCandidate(newSlug)
  }
})
</script>
