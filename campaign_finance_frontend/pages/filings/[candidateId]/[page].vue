<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useEventListener } from '@vueuse/core'
import { useFilingsStore } from '@/store/filings'

import NavBar from '@/components/NavBar.vue'
import AppFooter from '@/components/AppFooter.vue'
import ActiveHighlight from '@/components/ActiveHighlight.vue'

/* ─────────────────────────── logs ─────────────────────────── */
const LOG = (...args) => console.log('[QC:thumbs]', ...args)

/* ───────────────── routing / ids (early) ─────────────────── */
const route  = useRoute()
const router = useRouter()
const candId = computed(() => route.params.candidateId)
const pageNo = computed(() => Number(route.params.page || 1))

/* ───────────────────── store & helpers ───────────────────── */
const store = useFilingsStore()
function isThumbApproved (n) {
  if (typeof store?.isPageApproved === 'function') return !!store.isPageApproved(n)
  return !!store?.approvedPages?.[Number(n)]
}

/* ─────────────────────── DOM refs ────────────────────────── */
const pdfImg = ref(null)
const pdfViewport = ref(null)
const thumbsScroll = ref(null)

/* refs to each thumb node */
const thumbRefMap = new Map()
const setThumbRef = (n) => (el) => { if (el) thumbRefMap.set(n, el); else thumbRefMap.delete(n) }

/* ───────────── persist/restore scroll across remounts ───── */
const RESTORE_KEY = computed(() => `thumbsRestore:${candId.value}`)

function saveRestore (n, prevTop, keepOff) {
  try { sessionStorage.setItem(RESTORE_KEY.value, JSON.stringify({ n, prevTop, keepOff, t: Date.now() })) } catch {}
}
function readRestore () {
  try { const s = sessionStorage.getItem(RESTORE_KEY.value); return s ? JSON.parse(s) : null } catch { return null }
}
function clearRestore () {
  try { sessionStorage.removeItem(RESTORE_KEY.value) } catch {}
}
function offsetWithin (el, container) {
  let top = 0, node = el
  while (node && node !== container) { top += node.offsetTop || 0; node = node.offsetParent }
  return top
}
function attemptRestore (label = 'attempt') {
  const pending = readRestore()
  const container = thumbsScroll.value
  if (!pending) return true
  if (!container) { LOG('restore:', label, '→ no container'); return false }

  const comp = thumbRefMap.get(pending.n)
  const node = comp?.$el || comp
  if (!node) { LOG('restore:', label, '→ node not ready'); return false }

  const newOffset = offsetWithin(node, container)
  const target = (pending.keepOff != null) ? (pending.prevTop + (newOffset - pending.keepOff)) : pending.prevTop

  const before = container.scrollTop
  container.style.scrollBehavior = 'auto'
  container.scrollTop = target
  container.style.scrollBehavior = ''
  const after = container.scrollTop
  const ok = Math.abs(after - target) <= 1
  LOG('restore:', label, { n: pending.n, before, target, after, ok })
  if (ok) clearRestore()
  return ok
}

let restoreRaf = 0
function queueRestoreRaf () {
  cancelAnimationFrame(restoreRaf)
  restoreRaf = requestAnimationFrame(() => {
    if (!attemptRestore('raf')) queueRestoreRaf()
  })
}

/* watch container presence & finish pending restore */
watch(thumbsScroll, (nv, ov) => {
  LOG('thumbsScroll ref changed →', !!ov, '→', !!nv)
  if (nv) {
    try { nv.style.setProperty('overflow-anchor', 'none') } catch {}
    LOG('overflow-anchor set to none on thumbs container')
    if (readRestore()) queueRestoreRaf()
  }
})

/* scroll diagnostics */
useEventListener(thumbsScroll, 'scroll', () => {
  const el = thumbsScroll.value
  if (el) LOG('scrollTop=', el.scrollTop, 'scrollHeight=', el.scrollHeight)
}, { passive: true })

/* ───────────── image scale / zoom / rotation ───────────── */
const imgScale = ref(1)
const zoom     = ref(1)
const rotation = ref(0)
const hasUserZoomed = ref(false)

function syncScale () {
  if (!pdfImg.value) return
  const natW  = pdfImg.value.naturalWidth || 1
  const baseW = pdfImg.value.clientWidth || 1
  imgScale.value = (baseW * zoom.value) / natW
  LOG('syncScale →', { natW, baseW, zoom: zoom.value.toFixed(2), imgScale: imgScale.value.toFixed(3) })
}

function zoomIn  () { hasUserZoomed.value = true; zoom.value = Math.min(zoom.value + 0.1, 3) }
function zoomOut () { hasUserZoomed.value = true; zoom.value = Math.max(zoom.value - 0.1, 0.2) }
function rotateLeft  () { rotation.value = (rotation.value - 90 + 360) % 360 }
function rotateRight () { rotation.value = (rotation.value + 90) % 360 }

watch([zoom, rotation], async () => { await nextTick(); syncScale() })
watch(rotation, v => { store.currentRotation = v })

/* ─────────────── magnifier (hover lens) ──────────────── */
const magnifierOn  = ref(true)
const lensVisible  = ref(false)
const lensPower    = ref(2)
const lensDiameter = ref(180)
const lensScreenX  = ref(0)
const lensScreenY  = ref(0)
const bgW = ref(0), bgH = ref(0), bgX = ref(0), bgY = ref(0)

function onEnter () { if (magnifierOn.value) lensVisible.value = true }
function onLeave () { lensVisible.value = false }
function onMove (e) {
  if (!magnifierOn.value || !pdfImg.value) return
  const rect = pdfImg.value.getBoundingClientRect()
  if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) { lensVisible.value = false; return }
  lensVisible.value = true
  const x = Math.min(Math.max(e.clientX - rect.left, 0), rect.width)
  const y = Math.min(Math.max(e.clientY - rect.top , 0), rect.height)
  let nx = x / rect.width, ny = y / rect.height
  const r = ((rotation.value % 360) + 360) % 360
  let u = nx, v = ny
  if (r === 90)      { u = ny;     v = 1 - nx }
  else if (r === 180){ u = 1 - nx; v = 1 - ny }
  else if (r === 270){ u = 1 - ny; v = nx }
  const M = lensPower.value, radius = lensDiameter.value / 2
  bgW.value = rect.width * M; bgH.value = rect.height * M
  bgX.value = -(u * bgW.value - radius); bgY.value = -(v * bgH.value - radius)
  lensScreenX.value = e.clientX - radius; lensScreenY.value = e.clientY - radius
}
const lensStyle = computed(() => ({
  position: 'fixed',
  left: `${lensScreenX.value}px`,
  top: `${lensScreenY.value}px`,
  width: `${lensDiameter.value}px`,
  height: `${lensDiameter.value}px`,
  borderRadius: '50%',
  boxShadow: '0 0 0 2px rgba(0,0,0,.15), 0 8px 20px rgba(0,0,0,.25)',
  backgroundImage: store.pageImg ? `url(${store.pageImg})` : 'none',
  backgroundRepeat: 'no-repeat',
  backgroundSize: `${bgW.value}px ${bgH.value}px`,
  backgroundPosition: `${bgX.value}px ${bgY.value}px`,
  zIndex: 9999,
  pointerEvents: 'none',
  transform: `rotate(${rotation.value}deg)`,
}))

/* ─────────────── fit to viewport ──────────────── */
function fitToViewport () {
  if (!pdfViewport.value || !pdfImg.value) return
  const vp = pdfViewport.value.getBoundingClientRect()
  const natW = pdfImg.value.naturalWidth || 1
  const natH = pdfImg.value.naturalHeight || 1
  const r = ((rotation.value % 360) + 360) % 360
  const baseW = (r === 90 || r === 270) ? natH : natW
  const baseH = (r === 90 || r === 270) ? natW : natH
  const margin = 8
  const scaleW = (vp.width  - margin) / baseW
  const scaleH = (vp.height - margin) / baseH
  const next = Math.max(0.2, Math.min(3, Math.min(scaleW, scaleH)))
  const DEFAULT_INITIAL_ZOOM = 0.80
  zoom.value = Math.max(next, DEFAULT_INITIAL_ZOOM)
  LOG('fitToViewport →', { vpW: vp.width, vpH: vp.height, natW, natH, zoom: zoom.value.toFixed(2) })
}
async function onImgLoad () {
  LOG('onImgLoad fired')
  syncScale()
  if (!hasUserZoomed.value) {
    await nextTick()
    fitToViewport()
    await nextTick()
    syncScale()
  }
}

/* ───────────── click-drag panning (and warning fix) ───────────── */
const isPanning = ref(false)
const panStartX = ref(0), panStartY = ref(0)
const panScrollLeft = ref(0), panScrollTop = ref(0)

function onPanStart (e) {
  const el = pdfViewport.value; if (!el) return
  if (e.pointerType === 'mouse' && e.button !== 0) return
  e.preventDefault()
  isPanning.value = true
  panStartX.value = e.clientX; panStartY.value = e.clientY
  panScrollLeft.value = el.scrollLeft; panScrollTop.value = el.scrollTop
  const ct = e.currentTarget; if (ct?.setPointerCapture) { try { ct.setPointerCapture(e.pointerId) } catch {} }
  lensVisible.value = false
}
function onPanMove (e) {
  const el = pdfViewport.value; if (!el) return
  if (!isPanning.value) { onMove(e); return }
  if (e.pointerType === 'mouse' && e.buttons === 0) { onPanEnd(e); return }
  e.preventDefault()
  const dx = e.clientX - panStartX.value, dy = e.clientY - panStartY.value
  el.scrollLeft = panScrollLeft.value - dx; el.scrollTop = panScrollTop.value - dy
}
function onPanEnd (e) {
  if (!isPanning.value) return
  isPanning.value = false
  const ct = e?.currentTarget; if (ct?.releasePointerCapture) { try { ct.releasePointerCapture(e.pointerId) } catch {} }
}

/* ─────────────────── lifecycle / diagnostics ─────────────────── */
onMounted(() => {
  LOG('mounted with page=', pageNo.value, 'candidate=', candId.value)
  store.fetchPage(candId.value, pageNo.value)
})
watch(() => store.currentPageNum, (nv, ov) => LOG('currentPageNum', ov, '→', nv))
watch(() => store.totalPages, (nv, ov) => LOG('totalPages', ov, '→', nv))

/* ─────── re-run restore as the thumbs list repopulates ─────── */
watch(() => store.filesMap, (nv) => {
  LOG('filesMap changed; keys:', nv ? Object.keys(nv).length : 0)
})
const pageGroups = computed(() => {
  const groups = new Map()
  const files = store.filesMap || {}
  const total = Number(store.totalPages || 0)
  for (let n = 1; n <= total; n++) {
    const url = files[`page_${n}`] || store.thumbSrc(n)
    const key = docKeyFromUrl(url || '')
    if (!groups.has(key)) groups.set(key, { id: key, pages: [] })
    groups.get(key).pages.push(n)
  }
  const arr = Array.from(groups.values()).map(g => ({ ...g, first: g.pages[0] }))
  arr.sort((a, b) => a.first - b.first)
  const out = arr.map((g, i) => ({ ...g, alt: i % 2 === 1 }))
  LOG('pageGroups computed:', out.length, 'groups')
  return out
})
watch(pageGroups, () => { if (readRestore()) queueRestoreRaf() })

/* ───────────────────── style bindings ───────────────────── */
const rotationStyle = computed(() => ({
  transform: `rotate(${rotation.value}deg)`,
  transformOrigin: 'center center',
}))
const scaleStyle = computed(() => ({
  transform: `scale(${zoom.value})`,
  transformOrigin: 'center center',
}))

/* ───────────────────── navigation ───────────────────── */
async function goToPage (n) {
  const container = thumbsScroll.value
  const oldComp = thumbRefMap.get(n)
  const oldEl   = oldComp?.$el || oldComp
  const prevTop = container ? container.scrollTop : 0
  const keepOff = (container && oldEl) ? offsetWithin(oldEl, container) : null

  LOG('goToPage:start', { n, prevTop, keepOff, containerExists: !!container })
  saveRestore(n, prevTop, keepOff)

  await router.push(`/filings/${candId.value}/${n}`)
  await store.fetchPage(candId.value, n)
  await nextTick()

  // If Nuxt reused the same instance, try once here too.
  if (readRestore()) attemptRestore('post-fetch')
}

/* ───────────────── re-extract (unchanged) ──────────────── */
const isReextracting = ref(false)
const reextractErr = ref(null)
async function reextractWithRotation () {
  try {
    isReextracting.value = true
    reextractErr.value = null
    await $fetch('/api/finance/reextract', {
      method: 'POST',
      body: {
        slug: candId.value,
        global_page: pageNo.value,
        rotation: ((rotation.value % 360) + 360) % 360,
        overwrite_image: true,
        reset_validated: true
      }
    })
    rotation.value = 0
    hasUserZoomed.value = false
    await store.fetchPage(candId.value, pageNo.value)
    await nextTick()
    fitToViewport()
  } catch (e) {
    console.error(e)
    reextractErr.value = (e && (e.statusMessage || e.message)) || 'Re-extract failed'
  } finally {
    isReextracting.value = false
  }
}

/* ───────────── keyboard shortcuts (unchanged) ─────────── */
useEventListener('keydown', (e) => {
  const t = e.target
  const tag = t?.tagName ? String(t.tagName).toUpperCase() : ''
  const isEditable = (t && 'isContentEditable' in t && t.isContentEditable) || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
  if (isEditable) return
  if (e.key === 'Tab' && !e.shiftKey) { e.preventDefault(); store.selectNext() }
  if (e.key === 'Tab' &&  e.shiftKey) { e.preventDefault(); store.selectPrev() }
  if (e.key === 'Enter')              { store.toggleEdit?.() }
  if (e.key === '+')                  { zoomIn() }
  if (e.key === '-')                  { zoomOut() }
  if (e.key.toLowerCase() === 'r')    { rotateRight() }
})

/* ───────────────────────── cleanup ─────────────────────── */
onBeforeUnmount(() => {
  cancelAnimationFrame(restoreRaf)
  restoreRaf = 0
})

/* ───────────── sidebar grouping helpers (unchanged) ───── */
function docKeyFromUrl (url) {
  try {
    const p = new URL(url).pathname
    const oIdx = p.indexOf('/o/')
    const obj = decodeURIComponent(oIdx >= 0 ? p.slice(oIdx + 3) : p.slice(1))
    const fname = obj.slice(obj.lastIndexOf('/') + 1)
    const i = fname.indexOf('_page')
    return i >= 0 ? fname.slice(0, i) : fname
  } catch { return 'unknown' }
}
function hueForKey (key) { let h = 0; for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) % 360; return h }
function colorsForKey (key) { const h = hueForKey(key); return { color: `hsl(${h}, 65%, 50%)`, bg: `hsla(${h}, 65%, 50%, 0.12)` } }

/* ───────────── UI helper (unchanged) ───────────── */
function humanizeKey (k) {
  const map = {
    donor:'Donor', contributor:'Donor', payee:'Payee', name:'Name',
    date:'Date', address:'Address', city:'City', state:'State', zip:'ZIP',
    type:'Type', purpose:'Purpose', amount:'Amount', value:'Value', balance:'Balance',
    description:'Description'
  }
  return map[k] || k.replace(/_/g, ' ')
                   .split(' ')
                   .map(w => (w ? w[0].toUpperCase() + w.slice(1) : ''))
                   .join(' ')
}
</script>



<template>
  <v-app>
    <NavBar />
    <v-main>
      <v-container fluid>
        <v-row>
          <!-- ───────── Sidebar thumbnails (grouped by parent doc) ───────── -->
          <!-- ───────── Sidebar thumbnails (grouped by parent doc) ───────── -->
          <v-col cols="2" class="pt-4 thumbs-col">
            <div class="thumbs-scroll" ref="thumbsScroll">
              <div v-for="(g, gi) in pageGroups" :key="g.id" class="thumb-group">
                <div class="thumb-group__header" :class="{ alt: g.alt }">
                  <span class="thumb-group__title">Doc {{ gi + 1 }}</span>
                  <span class="thumb-group__range">p. {{ g.pages[0] }}–{{ g.pages[g.pages.length - 1] }}</span>
                </div>
                <div class="thumb-group__list">
                  <v-img
                      v-for="n in g.pages"
                      :key="n"
                      :src="store.thumbSrc(n)"
                      width="160" height="90" cover
                      class="mb-3 rounded cursor-pointer thumb-item"
                      :class="[
                        n === store.currentPageNum ? 'thumb-active' : 'elevation-1',
                        g.alt ? 'thumb-item--alt' : '',
                        isThumbApproved(n) ? 'thumb-approved' : ''
                      ]"
                      :ref="setThumbRef(n)"
                      @click="goToPage(n)"
                    />



                </div>
              </div>
            </div>
          </v-col>



          <!-- ───────── PDF viewport + controls ───────── -->
          <v-col cols="6" class="d-flex flex-column">
            <!-- controls -->
            <div class="d-flex align-center gap-2 mb-2">
              <v-btn icon size="small" title="Rotate Left"  @click="rotateLeft"><v-icon size="18">mdi-rotate-left</v-icon></v-btn>
              <v-btn icon size="small" title="Rotate Right" @click="rotateRight"><v-icon size="18">mdi-rotate-right</v-icon></v-btn>
              <v-btn icon size="small" title="Zoom In"       @click="zoomIn"><v-icon size="18">mdi-magnify-plus-outline</v-icon></v-btn>
              <v-btn icon size="small" title="Zoom Out"      @click="zoomOut"><v-icon size="18">mdi-magnify-minus-outline</v-icon></v-btn>
              <span class="ml-2 text-caption">{{ (zoom * 100).toFixed() }}%</span>
              <v-btn
                icon
                size="small"
                :color="magnifierOn ? 'primary' : undefined"
                :title="magnifierOn ? 'Magnifier: on' : 'Magnifier: off'"
                @click="magnifierOn = !magnifierOn"
              >
                <v-icon size="18">mdi-magnify</v-icon>
              </v-btn>

              <!-- NEW: persist rotation + re-extract -->
              <v-btn
                size="small"
                color="primary"
                class="ml-2"
                :loading="isReextracting"
                :disabled="isReextracting"
                title="Persist this rotation and re-extract this page"
                @click="reextractWithRotation"
              >
                Save rotation & re-extract
              </v-btn>
            </div>

            <div v-if="reextractErr" class="text-error text-caption mb-2">
              {{ reextractErr }}
            </div>

            <!-- viewport centers the page so rotation pivots visually around center -->
            <div
              ref="pdfViewport"
              class="pdf-viewport"
              :class="{ 'is-panning': isPanning }"
              @pointerdown="onPanStart"
              @pointermove="onPanMove"
              @pointerup="onPanEnd"
              @pointerleave="onPanEnd"
              @pointercancel="onPanEnd"
              @lostpointercapture="onPanEnd"
              @mousemove="onMove"
              @mouseenter="onEnter"
              @mouseleave="onLeave"
            >
              <div :style="rotationStyle" class="pdf-rotation">
                <div :style="scaleStyle" class="pdf-scale">
                  <!-- main page image -->
                  <img
                    ref="pdfImg"
                    :src="store.pageImg"
                    style="width:100%"
                    draggable="false"
                    @dragstart.prevent
                    @load="onImgLoad"
                  />

                  <!-- overlay rectangles -->
                  <ActiveHighlight
                    v-for="b in store.bboxes"
                    :key="b.id"
                    :bbox="b"
                    :scale="imgScale"
                    :active="b.id === store.currentId"
                    :approved="store.isApproved(b.id)"
                  />
                </div>
              </div>
            </div>
          </v-col>
<!-- ───────── Extraction table (independent scroll, inline editable) ───────── -->
<v-col cols="4" class="extract-col">
  <div class="extract-scroll">
    <v-table hover class="extract-table">
      <tbody>
        <tr
          v-for="row in store.rows"
          :key="row.id"
          :class="{
            'bg-yellow-lighten-4': row.id === store.currentId,
            'bg-green-lighten-5' : store.isApproved?.(row.id)
          }"
          @click="store.currentId = row.id"
          
        >
          <!-- Left cell (label/field) — editable -->
          <td
            class="text-grey-darken-2 editable"
            contenteditable
            tabindex="0"
            spellcheck="false"
            @click.stop
            @keydown.stop
            @keydown.enter.prevent="$event.currentTarget.blur()"
            @paste.prevent="document.execCommand('insertText', false, ($event.clipboardData || window.clipboardData).getData('text'))"
            @input="
              (row.label ? row.label = $event.currentTarget.innerText : row.field = $event.currentTarget.innerText);
              row._dirty = true
            "
          >
            {{ row.label || row.field }}
          </td>

          <!-- Right cell: show structured columns when present; otherwise single value -->
          <td>
            <!-- Structured object: show each key/value; values editable -->
            <template v-if="row.columns">
              <div
                v-for="(val, key) in row.columns"
                :key="key"
                class="kv"
                v-show="val !== '' && val !== null && val !== undefined"
              >
                <span class="k">{{ humanizeKey(key) }}:</span>
                <span
                  class="v editable"
                  contenteditable
                  tabindex="0"
                  spellcheck="false"
                  @click.stop
                  @keydown.stop
                  @keydown.enter.prevent="$event.currentTarget.blur()"
                  @paste.prevent="document.execCommand('insertText', false, ($event.clipboardData || window.clipboardData).getData('text'))"
                  @input="row.columns[key] = $event.currentTarget.innerText; row._dirty = true"
                >
                  {{ val }}
                </span>
              </div>
            </template>

            <!-- Fallback: a single editable value -->
            <template v-else>
              <div
                class="editable"
                contenteditable
                tabindex="0"
                spellcheck="false"
                @click.stop
                @keydown.stop
                @keydown.enter.prevent="$event.currentTarget.blur()"
                @paste.prevent="document.execCommand('insertText', false, ($event.clipboardData || window.clipboardData).getData('text'))"
                @input="row.value = $event.currentTarget.innerText; row._dirty = true"
              >
                {{ row.value }}
              </div>
            </template>
          </td>
        </tr>
      </tbody>
    </v-table>
  </div>

  <!-- Actions stay visible while the table scrolls -->
  <div class="extract-actions">
    <v-btn color="#38A169" @click="store.approvePage(candId.value)">
      Approve this page
    </v-btn>
    <v-btn color="#E53E3E" class="ml-4" @click="store.toggleEdit?.()">
      Edit
    </v-btn>
  </div>
</v-col>





        </v-row>
      </v-container>
    </v-main>
    <AppFooter />
    <!-- floating magnifier overlay -->
    <div v-if="lensVisible && magnifierOn" :style="lensStyle" class="magnifier"></div>
  </v-app>
</template>

<style scoped>
.cursor-pointer { cursor:pointer; }
.thumb-active   { outline:3px solid #3182CE; }

/* viewport centers content so rotation uses element center */
.pdf-viewport {
  position: relative;
  width: 100%;
  height: calc(100vh - 200px);
  overflow: auto;
  background: #fafafa;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: grab;
  touch-action: none; /* implement custom pan */
}

.pdf-viewport.is-panning {
  cursor: grabbing;
  user-select: none;
}

.pdf-rotation,
.pdf-scale {
  transform-origin: center center;
}

.magnifier { /* optional extra styles */ }

.thumb-group { margin-bottom: 12px; }
.thumb-group__header {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: #2d3748; /* gray-800 */
  padding: 6px 8px; border-left: 4px solid #e5e7eb; border-radius: 6px;
  margin-bottom: 8px; background: #f3f4f6; /* gray-100 */
}
.thumb-group__header.alt { background: #e5e7eb; /* gray-200 */ border-left-color: #d1d5db; /* gray-300 */ }
.thumb-group__title { font-weight: 600; }
.thumb-group__range { margin-left: auto; opacity: .7; }
.thumb-item { transition: transform .08s ease; border-left: 4px solid #e5e7eb; }
.thumb-item--alt { border-left-color: #d1d5db; }
.thumb-item:hover { transform: translateX(2px); }
.thumb-group__dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

/* Thumbnails: fixed height + internal scroll */
.thumbs-col {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 200px); /* match .pdf-viewport height */
  min-height: 0;
}
.thumbs-scroll {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  scroll-behavior: auto;
  -webkit-overflow-scrolling: touch;
  
  padding-right: 6px; 
}

.thumb-approved {
  border: 2px solid #38A169;
  box-shadow: inset 0 0 0 1px rgba(56, 161, 105, .35);
}

/* Remove the default left stripe when approved so borders don't stack */
.thumb-approved.thumb-item {
  border-left: none;
}

/* Right column: fixed height + internal scroll */
.extract-col {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 200px); /* matches .pdf-viewport height */
  min-height: 0;
}
.extract-scroll {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  -webkit-overflow-scrolling: touch;
  padding-right: 6px; /* room for scrollbar */
}
.extract-actions {
  flex: 0 0 auto;
  padding-top: 10px;
  background: transparent;
}

/* Key/value rows like in the screenshot */
.kv { display:flex; gap:.5rem; line-height:1.35; }
.kv .k { min-width:7rem; color:#4a5568; }
.kv .v { color:#1a202c; }

/* Inline editing affordances */
.editable { cursor:text; user-select:text; white-space:pre-wrap; min-width:8rem; min-height:30px; outline:none; }
.editable:focus { box-shadow: inset 0 0 0 2px #3b82f6; background:#ecfeff; }

.kv { display:flex; gap:.5rem; line-height:1.35; }
.kv .k { min-width:7rem; color:#4a5568; }
.kv .v { color:#1a202c; }

</style>
