# cloud_function/routes/finance_query.py
"""
Finance-query endpoints
──────────────────────────────────────────────────────────────────────────────
All read-only wrappers around pre-aggregated (or raw) Firestore data.

Collections
-----------
finance_summary        – per-candidate, per-cycle roll-ups
finance_contributions  – raw contribution rows
finance_expenditures   – raw expenditure rows
"""
from __future__ import annotations

import base64, logging
from datetime import datetime, timezone
from typing      import Dict, Optional, Tuple, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter, Query as FsQuery
from pydantic import BaseModel, Field

from cloud_function.utils.firestore_utils import get_firestore_client

logger      = logging.getLogger(__name__)
router      = APIRouter()
MAX_PAGE    = 200
CURSOR_SEP  = "§"         # unlikely in IDs – keeps the cursor short
# ───────────────────────────── helpers ──────────────────────────────────── #
def _latest_cycle(cycles: Dict[str, dict]) -> str:
    """Newest key – works because keys start with YYYY_."""
    return max(cycles) if cycles else ""

def _encode_cursor(ts: datetime, doc_id: str) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    blob = f"{ts.isoformat()}{CURSOR_SEP}{doc_id}".encode()
    return base64.urlsafe_b64encode(blob).decode()

def _decode_cursor(token: str) -> Tuple[datetime, str]:
    try:
        ts_str, doc_id = base64.urlsafe_b64decode(token.encode()).decode().split(CURSOR_SEP, 1)
        return datetime.fromisoformat(ts_str), doc_id
    except Exception as exc:
        raise HTTPException(400, "Invalid page_token") from exc
    

# ─────────────────────── state ▸ full-candidate list ────────────────────── #
# ─────────────────────── state ▸ full-candidate list ────────────────────── #
@router.get("/list/candidates")
async def list_candidates_for_state(
    state : str = Query(...,
        min_length=2, max_length=20,
        description="Two-letter postal code (e.g. KY)"
    ),
    limit : int = Query(500, ge=1, le=1000),
    client       = Depends(get_firestore_client),
):
    """
    Return **ALL** candidates that have finance data for *one* state.
    """
    st_norm = state.upper()

    try:
        docs = (
            client.collection("finance_summary")
                  .where("jurisdiction", "==", st_norm)
                  .order_by("candidate_name")
                  .limit(limit)
                  .stream()
        )

        # ⚠️  THE ONLY LINE THAT CHANGES IS THE FIRST ONE INSIDE THE DICT
        options = [{
            "slug"    : doc.id,                 # <- use canonical Firestore key
            "name"    : d.get("candidate_name"),
            "office"  : d.get("office"),
            "district": d.get("district"),
        } for doc in docs for d in [doc.to_dict()]]

        return {"hits": options, "state": st_norm}

    except Exception as exc:
        logger.exception("State-candidate list failed")
        raise HTTPException(500, f"Failed to list candidates: {exc}") from exc


# ───────────────────────────── look-ups ─────────────────────────────────── #
@router.get("/lookup/candidates")
async def lookup_candidates(
    q     : str = Query(..., min_length=2),
    state : str | None = Query(
        None, min_length=2, max_length=20,
        description="Optional two-letter postal code (e.g. KY)"
    ),
    limit : int = Query(15, ge=1, le=50),
    client       = Depends(get_firestore_client),
):
    """
    Prefix search on `candidate_slug` + fallback contains() on candidate_name.
    If `state` is supplied, results are limited to that jurisdiction.
    """
    q_norm = q.strip().lower()
    st_norm = state.upper() if state else None
    if not q_norm:
        return []

    try:
        results, seen = [], set()

        # fast prefix on slug ------------------------------------------------
        slug_docs = (
            client.collection("finance_summary")
                  .where(filter=FieldFilter("candidate_slug", ">=", q_norm))
                  .where(filter=FieldFilter("candidate_slug", "<",  q_norm + "\uf8ff"))
                  .limit(limit).stream()
        )
        for d in slug_docs:
            data = d.to_dict()
            if st_norm and data.get("jurisdiction") != st_norm:
                continue
            seen.add(d.id)
            results.append({
                "slug"    : d.id, 
                "name"    : data["candidate_name"],
                "office"  : data["office"],
                "district": data.get("district"),
            })
            if len(results) >= limit:
                return results

        # fallback contains() on name ---------------------------------------
        leftover = limit - len(results)
        if leftover:
            more = (
                client.collection("finance_summary")
                      .where(filter=FieldFilter("jurisdiction", "!=", None))
                      .limit(400).stream()
            )
            for d in more:
                if d.id in seen:
                    continue
                data = d.to_dict()
                if st_norm and data.get("jurisdiction") != st_norm:
                    continue
                if q_norm in data.get("candidate_name", "").lower():
                    results.append({
                        "slug"    : d.id, 
                        "name"    : data["candidate_name"],
                        "office"  : data["office"],
                        "district": data.get("district"),
                    })
                    if len(results) >= limit:
                        break
        print(results)
        return results
    except Exception as exc:
        logger.exception("Candidate lookup failed")
        raise HTTPException(500, f"Lookup failed: {exc}") from exc
# ------------------------------------------------------------------------- #
@router.get("/lookup/donors")
async def lookup_donors(
    q     : str = Query(..., min_length=2),
    state : str | None = Query(
        None, min_length=2, max_length=20,
        description="Optional two-letter postal code (e.g. KY)"
    ),
    limit : int = Query(20, ge=1, le=50),
    client       = Depends(get_firestore_client),
):
    """
    Prefix search on contributor_lc with optional jurisdiction filter.
    """
    q_norm  = q.strip().lower()
    st_norm = state.upper() if state else None
    if not q_norm:
        return []

    try:
        seen, donors = set(), []

        docs = (
            client.collection("finance_contributions")
                  .where(filter=FieldFilter("contributor_lc", ">=", q_norm))
                  .where(filter=FieldFilter("contributor_lc", "<",  q_norm + "\uf8ff"))
                  .limit(100).stream()
        )
        for d in docs:
            row  = d.to_dict()
            if st_norm and row.get("jurisdiction") != st_norm:
                continue
            name = row.get("contributor")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            donors.append({
                "name"  : name,
                "is_org": row.get("contributor_type") == "pac/org",
            })
            if len(donors) >= limit:
                break
        return donors
    except Exception as exc:
        logger.exception("Donor lookup failed")
        raise HTTPException(500, f"Lookup failed: {exc}") from exc
# ─────────────────────── candidate ▸ summary ──────────────────────────── #
class CycleSummary(BaseModel):
    candidate: Dict[str, str]
    cycle    : str
    metrics  : Dict[str, Any] 

@router.get("/candidates/{slug}/summary", response_model=CycleSummary)
async def candidate_summary(
    slug: str,
    cycle: Optional[str] = Query(None),
    client               = Depends(get_firestore_client),
):
    try:
        # 1️⃣  try “slug == document-id” (fast path)
        doc = client.collection("finance_summary").document(slug).get()

        # 2️⃣  fall back to field search (doc-ID mismatch)
        if not doc.exists:
            q = (client.collection("finance_summary")
                       .where("candidate_slug", "==", slug)
                       .limit(1).stream())
            doc = next(q, None)
            if not doc:
                raise HTTPException(404, "Candidate not found")

        data   = doc.to_dict()
        data.setdefault("candidate_slug", slug)
        cycles = data.get("cycles", {})
        if not cycles:
            raise HTTPException(404, "No finance data for candidate")

        cycle_key = cycle or max(cycles)          # newest cycle
        if cycle_key not in cycles:
            raise HTTPException(404, f"Cycle '{cycle_key}' not found")

        raw     = cycles[cycle_key]
        numeric_keys = {
            "raised_total", "spent_total", "individual_total",
            "pac_total", "txns",
            # add any other *numeric* keys you roll up in Firestore
        }
        metrics = {k: v for k, v in raw.items() if k in numeric_keys}

        return {
            "candidate": {  # (district already safe – you fixed that)
                "slug"        : data["candidate_slug"],
                "name"        : data["candidate_name"],
                "office"      : data["office"],
                "district"    : data.get("district") or "",
                "jurisdiction": data.get("jurisdiction"),
            },
            "cycle"  : cycle_key,
            "metrics": metrics,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Summary fetch failed")
        raise HTTPException(500, f"Failed to fetch summary: {exc}") from exc
# ─────────────────────── candidate ▸ top donors ───────────────────────── #
class _DonorAgg(BaseModel):
    name : str
    total: float
    txns : int

@router.get("/candidates/{slug}/top-donors")
async def top_donors(
    slug : str,
    cycle: Optional[str] = Query(None),
    limit: int = Query(15, ge=1, le=100),
    client               = Depends(get_firestore_client),
):
    try:
        cand_doc = client.collection("finance_summary").document(slug).get()
        if not cand_doc.exists:                     # fallback
            cand_iter = (client.collection("finance_summary")
                               .where("candidate_slug", "==", slug)
                               .limit(1).stream())
            cand_doc = next(cand_iter, None)
            if not cand_doc:
                raise HTTPException(404, "Candidate not found")

        cand      = cand_doc.to_dict()
        cycle_key = cycle or max(cand["cycles"])

        q = (client.collection("finance_contributions")
                    .where("made_to", "==", cand["candidate_name"])
                    .where("cycle_key", "==", cycle_key))

        donors: Dict[str, Dict[str, float | int | str]] = {}
        for d in q.stream():
            row  = d.to_dict()
            name = row["contributor"]
            agg  = donors.setdefault(
                name,
                {"name": name, "total": 0.0, "txns": 0,
                 "type": row.get("contributor_type", "individual")}
            )
            agg["total"] += float(row["amount"])
            agg["txns"]  += 1

        indiv = sorted((v for v in donors.values() if v["type"] != "pac/org"),
                       key=lambda x: x["total"], reverse=True)[:limit]
        pac   = sorted((v for v in donors.values() if v["type"] == "pac/org"),
                       key=lambda x: x["total"], reverse=True)[:limit]

        return {"individual": indiv, "pac": pac, "cycle": cycle_key}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Top donors fetch failed")
        raise HTTPException(500, f"Failed to fetch top donors: {exc}") from exc
# ─────────────────────── candidate ▸ spending ─────────────────────────── #
class SpendingSummary(BaseModel):
    slug        : str
    cycle       : str
    spent_total : float
    txns        : int

# ── helper (place near the top of routes/finance_query.py) ─────────────
def _coerce_amount(field) -> float:
    """
    Return a float regardless of whether `field` is a number, string or dict.
    Raises TypeError if it cannot interpret the value.
    """
    if isinstance(field, (int, float)):
        return float(field)
    if isinstance(field, str):
        return float(field.replace(",", "").replace("$", ""))
    if isinstance(field, dict):
        # Most‑common keys first; extend if your schema adds others.
        for key in ("value", "numeric_value", "amount", "raw"):
            if key in field:
                return _coerce_amount(field[key])
    raise TypeError(f"Unsupported amount shape: {field!r}")


# ── drop‑in replacement for existing endpoint ─────────────────────────
@router.get("/candidates/{slug}/spending")         # if you use APIRouter
async def candidate_spending(
    slug : str,
    cycle: Optional[str] = Query(None),
    client               = Depends(get_firestore_client),
):
    try:
        cand_doc = client.collection("finance_summary").document(slug).get()
        if not cand_doc.exists:
            cand_iter = (
                client.collection("finance_summary")
                      .where("candidate_slug", "==", slug)
                      .limit(1)
                      .stream()
            )
            cand_doc = next(cand_iter, None)
            if not cand_doc:
                raise HTTPException(404, "Candidate not found")

        cycle_key = cycle or max(cand_doc.to_dict()["cycles"])

        q = (
            client.collection("finance_expenditures")
                  .where("candidate_slug", "==", slug)
                  .where("cycle_key",     "==", cycle_key)
        )

        total, count = 0.0, 0
        for d in q.stream():
            row = d.to_dict()
            try:
                total += _coerce_amount(row["amount"])
                count += 1
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping txn %s with bad amount: %s", d.id, exc)

        return {
            "slug":        slug,
            "cycle":       cycle_key,
            "spent_total": total,
            "txns":        count,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Spending fetch failed")
        raise HTTPException(500, f"Failed to fetch spending: {exc}") from exc

# ───────────────────── raised-since (rolling window) ──────────────────── #
class RaisedSince(BaseModel):
    slug         : str
    since        : str  = Field(description="ISO date boundary, inclusive")
    raised_total : float
    txns         : int

@router.get("/candidates/{slug}/raised-since", response_model=RaisedSince)
async def raised_since(
    slug  : str,
    after : str = Query(..., description="YYYY-MM-DD (inclusive)"),
    client       = Depends(get_firestore_client),
):
    """Total $ + count of contributions *on/after* the given date."""
    try:
        try:
            boundary = datetime.fromisoformat(after).replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(400, "Parameter 'after' must be YYYY-MM-DD")

        q = (client.collection("finance_contributions")
                    .where("candidate_slug", "==", slug)
                    .where("receipt_date",   ">=", boundary))

        total, txns = 0.0, 0
        for d in q.stream():
            row   = d.to_dict()
            total += float(row["amount"])
            txns  += 1

        return {"slug": slug, "since": after,
                "raised_total": total, "txns": txns}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Raised-since fetch failed")
        raise HTTPException(500, f"Failed to fetch raised-since: {exc}") from exc
# ───────────────────── donor ▸ raw contributions ──────────────────────── #
@router.get("/donors/{name}/contributions")
async def donor_contributions(
    name : str,
    cycle: Optional[str] = Query(None),
    page_size : int      = Query(100, ge=1, le=MAX_PAGE),
    page_token: Optional[str] = Query(None),
    client                    = Depends(get_firestore_client),
):
    try:
        q: FsQuery = (client.collection("finance_contributions")
                             .where("contributor", "==", name))
        if cycle:
            q = q.where("cycle_key", "==", cycle)

        q = (q.order_by("receipt_date", direction=firestore.Query.DESCENDING)
               .order_by("__name__",      direction=firestore.Query.DESCENDING))

        if page_token:
            ts, doc_id = _decode_cursor(page_token)
            q = q.start_after({"receipt_date": ts, "__name__": doc_id})

        docs = list(q.limit(page_size).stream())
        rows = [d.to_dict() for d in docs]

        next_tok = (_encode_cursor(rows[-1]["receipt_date"], docs[-1].id)
                    if len(rows) == page_size else None)
        return {"rows": rows, "next_page_token": next_tok, "page_size": page_size}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Donor rows fetch failed")
        raise HTTPException(500, f"Failed to fetch contributions: {exc}") from exc
