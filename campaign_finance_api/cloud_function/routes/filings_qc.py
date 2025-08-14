# cloud_function/routes/filings_qc.py
from fastapi import APIRouter, Depends, HTTPException, Query
from google.cloud import firestore
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from cloud_function.utils.firestore_utils import get_firestore_client
from cloud_function.utils.storage_utils   import gs_to_http
import os
import re

import csv
import io
from datetime import datetime

from functools import partial

router = APIRouter(prefix="/finance")

# ---------------- helpers ----------------
def _split_address(addr: Optional[str]) -> Tuple[str, str, str, str]:
  if not addr:
    return "", "", "", ""
  m = re.search(r"^(.*?),\s*([^,]+)\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)?$", addr.strip())
  if m:
    street, city, state, z = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    return street, city, state, z
  return addr, "", "", ""

def _scalar(v: Any) -> Any:
  if v is None:
    return ""
  if isinstance(v, dict):
    for k in ("value", "amount", "numeric_value", "raw"):
      if k in v:
        return _scalar(v[k])
    return ""
  return v

def _row_label(row: Dict[str, Any]) -> str:
  for k in ("contributor", "payee", "creditor", "name"):
    if k in row:
      return str(_scalar(row[k]))
  return row.get("row_type", "")

def _signed_converter() -> partial:
  return partial(gs_to_http, signed=not bool(os.getenv("STORAGE_EMULATOR_HOST")))

def _build_global_index(client: firestore.Client, slug: str):
  """Return (index_map, files_map, total_pages)"""
  filings = (
      client.collection("filings")
            .where("candidate_slug", "==", slug)
            .order_by("scraped_at", direction=firestore.Query.DESCENDING)
            .stream()
  )
  filings = [f for f in filings]
  if not filings:
    raise HTTPException(404, "Candidate not found")

  index_map : Dict[int, Tuple[str, int]] = {}
  files_map : Dict[str, str]             = {}
  cursor = 0
  to_http = _signed_converter()
  for f in filings:
    d         = f.to_dict()
    doc_id    = f.id
    file_dict = d.get("files", {})
    pages_here = sorted(int(k.split("_")[1]) for k in file_dict if k.startswith("page_"))
    for p in pages_here:
      cursor += 1
      index_map[cursor] = (doc_id, p)
      files_map[f"page_{cursor}"] = to_http(file_dict[f"page_{p}"])
  return index_map, files_map, cursor

# ---------------- 1) list filings ----------------


@router.get("/filings")
async def list_filings(client: firestore.Client = Depends(get_firestore_client)):
  """
  One card per candidate_slug, aggregated across ALL of their filings:
    - totalPages : sum of pages across every filing doc for this slug
    - approved   : sum of pages with validated == True across every filing doc
    - name/office come from the most-recent filing (by scraped_at)
  """
  # 1) group all filings by candidate_slug and track most-recent metadata
  groups: Dict[str, Dict[str, Any]] = {}
  for doc in client.collection("filings").stream():
    d        = doc.to_dict() or {}
    slug     = d.get("candidate_slug") or "unknown"
    scraped  = d.get("scraped_at")
    files    = d.get("files", {}) or {}

    g = groups.get(slug)
    if not g:
      g = {
        "slug": slug,
        "latest": {"scraped_at": scraped, "name": d.get("candidate_name", ""), "office": d.get("office", ""), "doc_id": doc.id},
        "doc_ids": [doc.id],
        "total_pages": sum(1 for k in files if k.startswith("page_")),
      }
      groups[slug] = g
    else:
      # keep latest metadata
      if scraped and (g["latest"]["scraped_at"] is None or scraped > g["latest"]["scraped_at"]):
        g["latest"] = {"scraped_at": scraped, "name": d.get("candidate_name", ""), "office": d.get("office", ""), "doc_id": doc.id}
      # accumulate pages
      g["total_pages"] += sum(1 for k in files if k.startswith("page_"))
      g["doc_ids"].append(doc.id)

  # 2) count approved pages across ALL filing docs for each slug
  out: List[Dict[str, Any]] = []
  for slug, g in groups.items():
    approved_pages = 0
    for doc_id in g["doc_ids"]:
      pages_col = client.collection("filings").document(doc_id).collection("pages")
      approved_pages += sum(1 for _ in pages_col.where("validated", "==", True).stream())

    out.append({
      "id"         : slug,
      "race"       : g["latest"]["office"],
      "name"       : g["latest"]["name"],
      "totalPages" : int(g["total_pages"]),    # ← now ~30 instead of 4
      "approved"   : int(approved_pages),      # ← approved pages across all filings
      "doc_id"     : g["latest"]["doc_id"],    # keep for convenience if you need it elsewhere
    })

  return out

# ---------------- 2) get page ----------------
@router.get("/filings/{slug}/pages")
async def filing_page(
  slug : str,
  page : int = Query(1, ge=1),
  client: firestore.Client = Depends(get_firestore_client),
):
  index_map, files_map, total_pages = _build_global_index(client, slug)
  if page < 1 or page > total_pages:
    raise HTTPException(404, "Page out of range")
  doc_id, local_page = index_map[page]

  filing_ref = client.collection("filings").document(doc_id)
  filing_root = filing_ref.get().to_dict() or {}

  img = files_map[f"page_{page}"]

  # NEW: build a global-index → approved flag map for all thumbnails
  approved_map: Dict[int, bool] = {}
  for g_idx, (doc_i, local_i) in index_map.items():
    p_snap = client.document(f"filings/{doc_i}/pages/{local_i}").get()
    approved_map[g_idx] = bool((p_snap.to_dict() or {}).get("validated", False)) if p_snap.exists else False

  rows: List[Dict[str, Any]] = []
  page_doc = filing_ref.collection("pages").document(str(local_page)).get()
  if page_doc.exists:
    snaps = (
      page_doc.reference
              .collection("rows")
              .order_by("row_order")
              .stream()
    )
    for snap in snaps:
      row = snap.to_dict() or {}
      if row.get("row_type") == "meta":
        continue
      rt = row.get("row_type")
      cols: Dict[str, Any] = {}
      if rt == "contribution":
        street, city, state, z = _split_address(_scalar(row.get("address")))
        cols = {
          "date": _scalar(row.get("date")),
          "donor": _scalar(row.get("contributor")),
          "address": street, "city": city, "state": state, "zip": z,
          "type": _scalar(row.get("type")),
          "amount": _scalar(row.get("amount")),
        }
      elif rt == "expenditure":
        cols = {
          "date": _scalar(row.get("date")),
          "payee": _scalar(row.get("name") or row.get("payee")),
          "address": _scalar(row.get("address")),
          "purpose": _scalar(row.get("purpose")),
          "amount": _scalar(row.get("amount")),
        }
      elif rt == "in_kind":
        cols = {
          "date": _scalar(row.get("date")),
          "donor": _scalar(row.get("contributor")),
          "description": _scalar(row.get("description")),
          "value": _scalar(row.get("value")),
        }
      elif rt == "debt":
        cols = {
          "date": _scalar(row.get("date")),
          "creditor": _scalar(row.get("creditor")),
          "purpose": _scalar(row.get("purpose")),
          "balance": _scalar(row.get("balance")),
        }

      norm = None
      if rt == "contribution":
        street, city, state, _zip = _split_address(_scalar(row.get("address")))
        meta = filing_root.get("meta") or {}
        norm = {
          "Candidate_First_Last": _scalar(meta.get("candidate_name")),
          "Office": filing_root.get("office", ""),
          "Election_Year": filing_root.get("cycle", ""),
          "General_Or_Primary": "",
          "Donation_Date": _scalar(row.get("date")),
          "Donor_Name": _scalar(row.get("contributor")),
          "Business_Donor_Address": street,
          "Donor_City": city, "Donor_State": state,
          "Donation_Type": _scalar(row.get("type")),
          "Amount": _scalar(row.get("amount")),
          "Donor_Type": "", "Busines_Owner_Officer": "",
        }

      rows.append({
        "id"      : snap.id,
        "label"   : _row_label(row),
        "value"   : _scalar(row.get("amount") or row.get("value") or row.get("balance")),
        "bbox"    : row.get("bbox"),
        "approved": row.get("validated", False),
        "row_type": rt,
        "columns" : {k: v for k, v in cols.items() if v not in ("", None)},
        "norm"    : norm,
      })

  if not rows:
    for k, meta_field in (filing_root.get("meta") or {}).items():
      rows.append({
        "id"      : f"meta_{k}",
        "label"   : k.replace("_", " ").title(),
        "value"   : _scalar(meta_field),
        "bbox"    : (meta_field.get("bbox") if isinstance(meta_field, dict) else None),
        "approved": False,
      })

  return {
    "doc_id"       : doc_id,
    "page"         : page,
    "totalPages"   : total_pages,
    "pageImg"      : img,
    "rows"         : rows,
    "files"        : files_map,
    "approvedPages": approved_map,  # NEW: per-page approval flags
  }

# ---------------- 3) approve a single row (legacy) ----------------
class QCUpdate(BaseModel):
  doc_id  : str
  page    : int = Field(..., ge=1)
  row_id  : str
  approved: bool

@router.post("/qc/approve-row", status_code=204)
async def approve_row(body: QCUpdate, client: firestore.Client = Depends(get_firestore_client)):
  row_ref = client.document(f"filings/{body.doc_id}/pages/{body.page}/rows/{body.row_id}")
  if not row_ref.get().exists:
    raise HTTPException(404, "Row not found")
  row_ref.update({ "validated": body.approved, "validated_at": SERVER_TIMESTAMP })

# ---------------- 4) update a row (legacy) ----------------
class RowEdit(BaseModel):
  doc_id: str
  page: int = Field(..., ge=1)
  row_id: str
  value: Any
  validated: Optional[bool] = None

@router.post("/qc/update-row", status_code=204)
async def update_row(body: RowEdit, client: firestore.Client = Depends(get_firestore_client)):
  row_ref = client.document(f"filings/{body.doc_id}/pages/{body.page}/rows/{body.row_id}")
  if not row_ref.get().exists:
    raise HTTPException(404, "Row not found")
  data = {"value": body.value, "updated": SERVER_TIMESTAMP}
  if body.validated is not None:
    data["validated"] = body.validated
    data["validated_at"] = SERVER_TIMESTAMP
  row_ref.update(data)

# ---------------- 5) NEW: approve page (with optional edits) -------
class RowPatch(BaseModel):
  row_id: str
  value: Optional[Any] = None
  columns: Optional[Dict[str, Any]] = None
  approved: Optional[bool] = None

class PageApprovePayload(BaseModel):
  page: int = Field(..., ge=1)        # global page index
  approved: bool = True               # approve even if rows is empty
  rows: Optional[List[RowPatch]] = None

@router.post("/filings/{slug}/pages/approve")
async def approve_page(
  slug: str,
  payload: PageApprovePayload,
  client: firestore.Client = Depends(get_firestore_client),
):
  index_map, _, total_pages = _build_global_index(client, slug)
  if payload.page < 1 or payload.page > total_pages:
    raise HTTPException(404, "Page out of range")
  doc_id, local_page = index_map[payload.page]

  batch = client.batch()

  # apply row edits if any
  for item in (payload.rows or []):
    row_ref = client.document(f"filings/{doc_id}/pages/{local_page}/rows/{item.row_id}")
    if not row_ref.get().exists:
      continue
    patch: Dict[str, Any] = { "updated": SERVER_TIMESTAMP }

    if item.value is not None:
      patch["value"] = item.value

    if item.columns:
      # Map common column keys to underlying fields
      col = item.columns
      if "date" in col:        patch["date"] = col["date"]
      if "donor" in col:
        patch["contributor"] = col["donor"]
        patch["name"] = col["donor"]
      if "payee" in col:
        patch["payee"] = col["payee"]
        patch["name"]  = col["payee"]
      if "creditor" in col:    patch["creditor"] = col["creditor"]
      if "address" in col:     patch["address"] = col["address"]
      if "city" in col:        patch["city"] = col["city"]
      if "state" in col:       patch["state"] = col["state"]
      if "zip" in col:         patch["zip"] = col["zip"]
      if "type" in col:        patch["type"] = col["type"]
      if "purpose" in col:     patch["purpose"] = col["purpose"]
      if "amount" in col:      patch["amount"] = col["amount"]
      if "value" in col:       patch["value"]  = col["value"]
      if "balance" in col:     patch["balance"] = col["balance"]
      if "description" in col: patch["description"] = col["description"]

    if item.approved is not None:
      patch["validated"] = bool(item.approved)
      patch["validated_at"] = SERVER_TIMESTAMP

    batch.update(row_ref, patch)

  # mark the page approved at the page level (even when no row edits)
  page_ref = client.document(f"filings/{doc_id}/pages/{local_page}")
  if page_ref.get().exists:
    batch.set(page_ref, { "validated": bool(payload.approved), "validated_at": SERVER_TIMESTAMP }, merge=True)
  else:
    batch.set(page_ref, { "validated": bool(payload.approved), "validated_at": SERVER_TIMESTAMP })

  batch.commit()
  return { "ok": True, "doc_id": doc_id, "local_page": local_page }




# ---------- Normalization helpers (contributions only) ----------
NORMALIZED_HEADERS = [
  "Candidate_First_Last",
  "Office",
  "Election_Year",
  "General_Or_Primary",
  "Donation_Date",
  "Donor_Name",
  "Business_Donor_Address",
  "Donor_City",
  "Donor_State",
  "Donation_Type",
  "Amount",
  "Donor_Type",
  "Busines_Owner_Officer",
]

def _latest_filing_docs_by_slug(client: firestore.Client):
  """Return dict slug -> (doc_id, filing_root dict) for the most recent filing per slug."""
  latest: Dict[str, Tuple[str, Dict[str, Any]]] = {}
  for doc in client.collection("filings").stream():
    d    = doc.to_dict() or {}
    slug = d.get("candidate_slug") or "unknown"
    cur  = latest.get(slug)
    if cur is None or (d.get("scraped_at") and d["scraped_at"] > (cur[1].get("scraped_at") if cur[1] else None)):
      latest[slug] = (doc.id, d)
  return latest

def _iter_normalized_contrib_rows(client: firestore.Client, doc_id: str, filing_root: Dict[str, Any]):
  """Yield dicts matching NORMALIZED_HEADERS for contributions in a filing doc."""
  # derive context
  meta   = filing_root.get("meta") or {}
  office = filing_root.get("office", "")
  cycle  = filing_root.get("cycle", "")
  cand_name = _scalar(meta.get("candidate_name"))

  # iterate all pages -> rows
  pages_ref = client.collection("filings").document(doc_id).collection("pages")
  for page_snap in pages_ref.stream():
    rows_ref = page_snap.reference.collection("rows")
    for row_snap in rows_ref.stream():
      row = row_snap.to_dict() or {}
      if row.get("row_type") != "contribution":
        continue

      street, city, state, _zip = _split_address(_scalar(row.get("address")))
      out = {
        "Candidate_First_Last"  : cand_name,
        "Office"                : office,
        "Election_Year"         : cycle,
        "General_Or_Primary"    : "",
        "Donation_Date"         : _scalar(row.get("date")),
        "Donor_Name"            : _scalar(row.get("contributor")),
        "Business_Donor_Address": street,
        "Donor_City"            : city,
        "Donor_State"           : state,
        "Donation_Type"         : _scalar(row.get("type")),
        "Amount"                : _scalar(row.get("amount")),
        "Donor_Type"            : "",
        "Busines_Owner_Officer" : "",
      }
      yield out

def _write_csv(rows_iter):
  buf = io.StringIO()
  w   = csv.DictWriter(buf, fieldnames=NORMALIZED_HEADERS, extrasaction="ignore")
  w.writeheader()
  for r in rows_iter:
    w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in NORMALIZED_HEADERS})
  return buf.getvalue()

# ---------- per-candidate export ----------
@router.get("/filings/{slug}/export")
async def export_candidate_csv(
  slug: str,
  format: str = "csv",
  client: firestore.Client = Depends(get_firestore_client),
):
  latest = _latest_filing_docs_by_slug(client)
  if slug not in latest:
    raise HTTPException(404, "Candidate not found")
  doc_id, filing_root = latest[slug]
  if format != "csv":
    raise HTTPException(400, "Only CSV is supported")

  csv_text = _write_csv(_iter_normalized_contrib_rows(client, doc_id, filing_root))
  from fastapi import Response
  return Response(
    content=csv_text,
    media_type="text/csv; charset=utf-8",
    headers={
      "Content-Disposition": f'attachment; filename="{slug}.csv"'
    }
  )

# ---------- export all candidates (latest filing per slug) ----------
from fastapi import Response

@router.get("/filings/export-all")
async def export_all_csv(
  format: str = "csv",
  client: firestore.Client = Depends(get_firestore_client),
):
  if format != "csv":
    raise HTTPException(400, "Only CSV is supported")

  latest = _latest_filing_docs_by_slug(client)

  def rows():
    for slug, (doc_id, filing_root) in latest.items():
      yield from _iter_normalized_contrib_rows(client, doc_id, filing_root)

  csv_text = _write_csv(rows())
  stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
  return Response(
    content=csv_text,
    media_type="text/csv; charset=utf-8",
    headers={
      "Content-Disposition": f'attachment; filename="all_candidates_{stamp}.csv"'
    }
  )
