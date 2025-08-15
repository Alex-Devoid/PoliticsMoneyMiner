<script setup>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useFilingsStore } from '@/store/filings'
import NavBar    from '@/components/NavBar.vue'
import AppFooter from '@/components/AppFooter.vue'

const router  = useRouter()
const filings = useFilingsStore()

onMounted(() => filings.fetchList())

function openCandidate (cand) {
  router.push(`/filings/${cand.id}/1`)
}

/* helpers: tolerate either `approved` or `approvedPages` from backend */
function approvedPages (cand) {
  const a = cand?.approved ?? cand?.approvedPages ?? 0
  return Number.isFinite(a) ? a : 0
}
function totalPages (cand) {
  const t = cand?.totalPages ?? 0
  return Number.isFinite(t) ? t : 0
}
function pctApproved (cand) {
  const a = approvedPages(cand)
  const t = totalPages(cand)
  return t > 0 ? Math.round((a / t) * 100) : 0
}
function chipColor (cand) {
  const a = approvedPages(cand)
  const t = totalPages(cand)
  return a === t && t > 0 ? '#38A169' : '#E53E3E'
}

/* -------- CSV downloads ---------- */
function downloadAllCsv () {
  // opens a file download; no auth headers needed since it’s proxied by Nuxt
  window.open('/api/filings/export-all', '_blank')
}
function downloadCandidateCsv (cand, e) {
  // don’t trigger row navigation
  if (e && typeof e.stopPropagation === 'function') e.stopPropagation()
  window.open(`/api/filings/export/${encodeURIComponent(cand.id)}`, '_blank')
}
</script>

<template>
  <v-app>
    <NavBar />
    <v-main>
      <v-container>

        <div class="d-flex align-center justify-space-between mb-6">
          <h1 class="text-h5">Filings Overview</h1>
          <v-btn color="primary" variant="flat" @click="downloadAllCsv">
            Download all (CSV)
          </v-btn>
        </div>

        <!-- loop races -->
        <template v-for="race in filings.raceGroups" :key="race.name">
          <v-sheet
            height="32"
            color="#2D3748"
            class="d-flex align-center px-4 mb-2 rounded"
          >
            <span class="text-white">{{ race.name }}</span>
          </v-sheet>

          <v-sheet
            v-for="cand in race.candidates"
            :key="cand.id"
            class="d-flex align-center justify-space-between px-4 mb-2 rounded cursor-pointer"
            height="56"
            elevation="1"
            @click="openCandidate(cand)"
          >
            <span class="text-body-2 text-medium-emphasis">{{ cand.name }}</span>

            <div class="d-flex align-center" style="min-width: 320px;">
              <!-- compact progress bar -->
              <v-progress-linear
                :model-value="pctApproved(cand)"
                height="6"
                rounded
                class="mr-3"
                style="width: 140px;"
              />
              <!-- pages approved / total -->
              <v-chip :color="chipColor(cand)" size="small" variant="elevated" class="mr-3">
                {{ approvedPages(cand) }} / {{ totalPages(cand) }}
              </v-chip>

              <!-- per-candidate CSV -->
              <v-btn size="small" variant="outlined" @click="(e) => downloadCandidateCsv(cand, e)">
                Download CSV
              </v-btn>
            </div>
          </v-sheet>
        </template>

      </v-container>
    </v-main>
    <AppFooter />
  </v-app>
</template>

<style scoped>
.cursor-pointer { cursor: pointer; }
</style>
