// scripts/makeMock.js
// -----------------------------------------------------------------------------
// Usage:  node scripts/makeMock.js <file.pdf>
//
// Produces two JSON files Nuxt already consumes:
//
//   mock/filings-overview.json     ← high-level candidate list
//   mock/jcerullo-pages.json       ← one node per PDF page
// -----------------------------------------------------------------------------

import fs from 'fs'
import { getDocument }   from 'pdfjs-dist/legacy/build/pdf.mjs'
import { createCanvas }  from '@napi-rs/canvas'

/* ─── CLI arg guard ───────────────────────────────────────────────────────── */
const [, , pdfPath] = process.argv
if (!pdfPath) {
  console.error('❌  Usage: node scripts/makeMock.js <file.pdf>')
  process.exit(1)
}
if (!fs.existsSync(pdfPath)) {
  console.error(`❌  File not found: ${pdfPath}`)
  process.exit(1)
}

/* ─── helper: rasterise 1 page → PNG buffer ──────────────────────────────── */
async function renderPageToPng (pdfDoc, pageNum, scale = 1.4) {
  const page     = await pdfDoc.getPage(pageNum)
  const viewport = page.getViewport({ scale })
  const canvas   = createCanvas(viewport.width, viewport.height)
  const ctx      = canvas.getContext('2d')

  await page.render({ canvasContext: ctx, viewport }).promise
  return canvas.toBuffer('image/png')
}

/* ─── 1. Open the PDF (as Uint8Array) ────────────────────────────────────── */
const buf     = await fs.promises.readFile(pdfPath)
const pdfDoc  = await getDocument({ data: new Uint8Array(buf) }).promise
const nPages  = pdfDoc.numPages           // = 7 for test_report.pdf

/* ─── 2. HARD-CODED extraction results (your JSON pasted here) ───────────── */

const report_meta = {
  candidate_name       : 'Jared Cerullo',
  address              : '6316 S Madison St',
  city_zip             : 'Wichita, KS 67216',
  county               : 'Sedgwick',
  office_sought        : 'City of Wichita – City Council',
  district             : '3',
  cash_start           : '$2,000.00',
  total_contributions  : '$10,811.20',
  total_expenditures   : '$3,463.36',
  cash_end             : '$9,347.84'
}

// schedule A – 20 contributions
const scheduleA = [
  { contributor:'Regency 21, LLC',                 amount:'$500.00',  date:'2021-06-29' },
  { contributor:'Wichita Crossing, LLC',           amount:'$500.00',  date:'2021-06-29' },
  { contributor:'Main Street Partners, LLC',       amount:'$500.00',  date:'2021-06-29' },
  { contributor:'J Russell Communities LLC',       amount:'$500.00',  date:'2021-06-30' },
  { contributor:'Jay Russell Devel & Mgmt, Inc.',  amount:'$500.00',  date:'2021-06-30' },
  { contributor:'Tow Service, Inc.',               amount:'$500.00',  date:'2021-07-01' },
  { contributor:'Ken\'s Auto Tow, Inc.',           amount:'$500.00',  date:'2021-07-01' },
  { contributor:'Reliable Towing of Wichita, LLC', amount:'$500.00',  date:'2021-07-01' },
  { contributor:'Don Sherman',                     amount:'$250.00',  date:'2021-07-06' },
  { contributor:'Brandy Miller',                   amount:'$250.00',  date:'2021-07-07' },
  { contributor:'Joshua Kippenberger',             amount:'$500.00',  date:'2021-05-20' },
  { contributor:'Summit Holdings',                 amount:'$500.00',  date:'2021-07-12' },
  { contributor:'Marketplace Properties',          amount:'$500.00',  date:'2021-07-14' },
  { contributor:'Max A Cohen Trust',               amount:'$500.00',  date:'2021-05-26' },
  { contributor:'Sixty-four Hundred LLC',          amount:'$500.00',  date:'2021-05-26' },
  { contributor:'Ferris Consulting',               amount:'$250.00',  date:'2021-07-27' },
  { contributor:'Expert Auto Care',                amount:'$500.00',  date:'2021-07-27' },
  { contributor:'James Ludwig',                    amount:'$400.00',  date:'2021-01-20' },
  { contributor:'Tom & Gerry Winters',             amount:'$500.00',  date:'2021-01-25' },
  { contributor:'Gerald Blood',                    amount:'$200.00',  date:'2021-02-10' }
]

// schedule C – 15 expenditures
const scheduleC = [
  { payee:'Walmart',                purpose:'Office Supplies',            amount:'$19.27',  date:'2021-01-25' },
  { payee:'Le Monde Cafe',          purpose:'Campaign Lunch',             amount:'$30.16',  date:'2021-02-11' },
  { payee:'Julia Burton',           purpose:'Website Design',             amount:'$260.00', date:'2021-04-20' },
  { payee:'Mike Carroll',           purpose:'Campaign Photos',            amount:'$75.00',  date:'2021-05-04' },
  { payee:'City Blue Print',        purpose:'Stationery + Envelopes',     amount:'$538.68', date:'2021-05-18' },
  { payee:'Julia Burton',           purpose:'Web Hosting',                amount:'$731.01', date:'2021-05-26' },
  { payee:'Tight wrapz Print Shop', purpose:'Palm Cards',                 amount:'$300.00', date:'2021-06-23' },
  { payee:'Facebook',               purpose:'Social Media Ads',           amount:'$20.00',  date:'2021-07-06' },
  { payee:'Facebook',               purpose:'Social Media Ads',           amount:'$10.00',  date:'2021-07-07' },
  { payee:'Tight wrapz Print Shop', purpose:'Campaign Banners',           amount:'$100.00', date:'2021-07-09' },
  { payee:'Facebook',               purpose:'Social Media Ads',           amount:'$15.00',  date:'2021-07-12' },
  { payee:'Sedgwick County Election',purpose:'Voter Information',         amount:'$61.31',  date:'2021-07-20' },
  { payee:'Facebook',               purpose:'Social Media Ads',           amount:'$5.00',   date:'2021-07-20' },
  { payee:'Tight wrapz Print Shop', purpose:'Yard Signs',                 amount:'$1,061.56',date:'2021-07-21' },
  { payee:'Paypal',                 purpose:'Transaction fees',           amount:'$36.37',  date:'2021-07-22' }
]

/* helper to build a row object (bbox null → reviewers must drag-select) */
function row (id, label, value) {
  return { id, label, value, bbox:null, approved:false }
}

/* ─── 3. Map hard-coded data → page-level rows ───────────────────────────── */

// Page-1 summary
const page1Rows = [
  row('meta-cand' , 'Candidate Name'  , report_meta.candidate_name),
  row('meta-addr' , 'Address'         , report_meta.address),
  row('meta-city' , 'City / ZIP'      , report_meta.city_zip),
  row('meta-county', 'County'         , report_meta.county),
  row('meta-office', 'Office Sought'  , report_meta.office_sought),
  row('meta-dist' , 'District'        , report_meta.district),
  row('meta-cash0', 'Cash Start'      , report_meta.cash_start),
  row('meta-contri','Contributions'   , report_meta.total_contributions),
  row('meta-exp'  , 'Expenditures'    , report_meta.total_expenditures),
  row('meta-cash1', 'Cash End'        , report_meta.cash_end)
]

// split schedule A across page-2 and page-3 (10 rows each)
const page2Rows = scheduleA.slice(0,10).map((c,i)=>row(`a-${i+1}`, c.contributor, `${c.date} – ${c.amount}`))
const page3Rows = scheduleA.slice(10).map((c,i)=>row(`a-${i+11}`,c.contributor, `${c.date} – ${c.amount}`))

// split schedule C across page-4 and page-5 (8 rows | 7 rows)
const page4Rows = scheduleC.slice(0,8).map((e,i)=>row(`c-${i+1}`, e.payee, `${e.date} – ${e.amount}`))
const page5Rows = scheduleC.slice(8).map((e,i)=>row(`c-${i+9}`, e.payee, `${e.date} – ${e.amount}`))

// keep pages 6-7 blank so QC still renders
const extractedByPage = {
  1: page1Rows,
  2: page2Rows,
  3: page3Rows,
  4: page4Rows,
  5: page5Rows,
  6: [],
  7: []
}

/* ─── 4. Build page objects + rasterise thumbnails ───────────────────────── */
const pages = []

for (let i = 1; i <= nPages; i++) {
  const pngBuf = await renderPageToPng(pdfDoc, i)

  pages.push({
    page       : i,
    pageImg    : `data:image/png;base64,${pngBuf.toString('base64')}`,
    rows       : extractedByPage[i] ?? [],
    totalPages : nPages,
  })
}

/* ─── 5. One-candidate overview list ─────────────────────────────────────── */
const overview = [{
  id        : 'jcerullo',
  race      : 'City Council — District 3',
  name      : report_meta.candidate_name,
  totalPages: nPages,
  approved  : 0,
}]

/* ─── 6. Persist JSON files ──────────────────────────────────────────────── */
await fs.promises.mkdir('mock', { recursive:true })
await fs.promises.writeFile('mock/filings-overview.json',
  JSON.stringify(overview, null, 2))
await fs.promises.writeFile('mock/jcerullo-pages.json',
  JSON.stringify(pages, null, 2))

console.log('✅  Mock data written:')
console.log('   • mock/filings-overview.json')
console.log('   • mock/jcerullo-pages.json')
