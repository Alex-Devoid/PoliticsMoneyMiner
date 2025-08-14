# cloud_function/routes/finance_query.py
"""
Finance-query endpoints
───────────────────────────────────────────────────────────────────────────────
Thin wrappers around pre-aggregated documents in Firestore.

Collections
-----------
finance_summary        – per-candidate, per-cycle roll-ups
finance_contributions  – raw contribution rows
finance_expenditures   – raw expenditure rows  (unused here)

Routes
------
GET /finance/lookup/candidates?q=
GET /finance/lookup/donors?q=
GET /finance/candidates/{slug}/summary?cycle=
GET /finance/candidates/{slug}/top-donors?cycle=&limit=
GET /finance/donors/{name}/contributions?cycle=&page_size=&page_token=
"""
from __future__ import annotations

from fastapi import APIRouter, Query, HTTPException, Depends
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter, Query as FsQuery
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import base64, json, logging

from cloud_function.utils.firestore_utils import get_firestore_client

logger   = logging.getLogger(__name__)
router   = APIRouter()
MAX_PAGE = 200                       # hard cap for paginated endpoints
CURSOR_SEP = "§"                     # unlikely in IDs; keeps cursor compact


# ─────────────────────────────── utils ──────────────────────────────────── #
def encode_cursor(receipt_date: datetime, doc_id: str) -> str:
    """Date + doc-id → compact url-safe cursor."""
    blob = f"{receipt_date.isoformat()}{CURSOR_SEP}{doc_id}".encode()
    return base64.urlsafe_b64encode(blob).decode()

def decode_cursor(token: str) -> Tuple[datetime, str]:
    """Reverse of `encode_cursor`."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        ts_str, doc_id = decoded.split(CURSOR_SEP, 1)
        return datetime.fromisoformat(ts_str), doc_id
    except Exception as exc:
        raise HTTPException(400, "Invalid page_token") from exc


def latest_cycle(cycles: Dict[str, dict]) -> str:
    """Return the newest cycle key (simple max works with YYYY_PRIMARY / GENERAL)."""
    return max(cycles) if cycles else ""


# ───────────────────────────── lookup -- candidates ────────────────────── #
@router.get("/lookup/candidates")
async def lookup_candidates(
    q: str = Query(..., min_length=2, description="Name or slug (fuzzy)"),
    limit: int = Query(15, ge=1, le=50),
    client      = Depends(get_firestore_client),
):
    """
    Simple prefix search on `candidate_slug`, with fallback to contains() on name.
    """
    q_norm = q.strip().lower()
    if not q_norm:
        return []

    try:
        # 1) Prefix on slug  (fast, uses indexes)
        slug_docs = (
            client.collection("finance_summary")
                  .where(filter=FieldFilter("candidate_slug", ">=", q_norm))
                  .where(filter=FieldFilter("candidate_slug", "<",  q_norm + "\uf8ff"))
                  .limit(limit)
                  .stream()
        )
        seen, results = set(), []
        for d in slug_docs:
            data = d.to_dict()
            seen.add(d.id)
            results.append({
                "slug"    : data["candidate_slug"],
                "name"    : data["candidate_name"],
                "office"  : data["office"],
                "district": data.get("district")
            })
            if len(results) >= limit:
                return results

        # 2) Fallback contains() on name (may scan, but small cardinality)
        leftovers = limit - len(results)
        if leftovers:
            more = (
                client.collection("finance_summary")
                      .where(filter=FieldFilter("jurisdiction", "!=", None))  # forces index use
                      .limit(400)           # safety – collection O(hundreds)
                      .stream()
            )
            for d in more:
                if d.id in seen:
                    continue
                data = d.to_dict()
                if q_norm in data.get("candidate_name", "").lower():
                    results.append({
                        "slug": data["candidate_slug"],
                        "name": data["candidate_name"],
                        "office": data["office"],
                        "district": data.get("district")
                    })
                    if len(results) >= limit:
                        break
        return results

    except Exception as exc:
        logger.exception("Candidate lookup failed")
        raise HTTPException(500, f"Lookup failed: {exc}") from exc


# ───────────────────────────── lookup -- donors ───────────────────────── #
@router.get("/lookup/donors")
async def lookup_donors(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=50),
    client      = Depends(get_firestore_client),
):
    """
    Prefix search on `contributor` in `finance_contributions`.
    Distinct donor names, plus `is_org` heuristic from `contributor_type`.
    """
    q_norm = q.strip().lower()
    seen, donors = set(), []

    try:
        docs = (
            client.collection("finance_contributions")
                  .where(filter=FieldFilter("contributor_lc", ">=", q_norm))
                  .where(filter=FieldFilter("contributor_lc", "<",  q_norm + "\uf8ff"))
                  .limit(100)       # read-cap; we dedupe client-side
                  .stream()
        )
        for d in docs:
            data = d.to_dict()
            name = data.get("contributor")
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            donors.append({
                "name"  : name,
                "is_org": data.get("contributor_type") == "pac/org"
            })
            if len(donors) >= limit:
                break
        return donors

    except Exception as exc:
        logger.exception("Donor lookup failed")
        raise HTTPException(500, f"Lookup failed: {exc}") from exc


# ───────────────────── candidate ▸ summary (single cycle) ─────────────── #
class CycleSummary(BaseModel):
    candidate: Dict[str, str]
    cycle: str
    metrics: Dict[str, float | int]

@router.get("/candidates/{slug}/summary", response_model=CycleSummary)
async def candidate_summary(
    slug: str,
    cycle: Optional[str] = Query(None),
    client                = Depends(get_firestore_client),
):
    try:
        doc = client.collection("finance_summary").document(slug).get()
        if not doc.exists:
            raise HTTPException(404, "Candidate not found")

        data = doc.to_dict()
        cycles = data.get("cycles", {})
        if not cycles:
            raise HTTPException(404, "No finance data for candidate")

        cycle_key = cycle or latest_cycle(cycles)
        if cycle_key not in cycles:
            raise HTTPException(404, f"Cycle '{cycle_key}' not found")

        return {
            "candidate": {
                "slug"     : data["candidate_slug"],
                "name"     : data["candidate_name"],
                "office"   : data["office"],
                "district" : data.get("district"),
                "jurisdiction": data.get("jurisdiction"),
            },
            "cycle":   cycle_key,
            "metrics": cycles[cycle_key],
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Summary fetch failed")
        raise HTTPException(500, f"Failed to fetch summary: {exc}") from exc


# ─────────────────── candidate ▸ top donors (aggregate) ────────────────── #
class _DonorAgg(BaseModel):
    name : str
    total: float = Field(..., examples=[2500.00])
    txns : int

@router.get("/candidates/{slug}/top-donors")
async def top_donors(
    slug: str,
    cycle: Optional[str] = Query(None),
    limit: int = Query(15, ge=1, le=100),
    client                = Depends(get_firestore_client),
):
    """
    Aggregate contributions for the given candidate & cycle.
    """
    try:
        cand_doc = client.collection("finance_summary").document(slug).get()
        if not cand_doc.exists:
            raise HTTPException(404, "Candidate not found")

        cand = cand_doc.to_dict()
        cycle_key = cycle or latest_cycle(cand["cycles"])

        # Stream contributions – typically O(thousands)
        q = (
            client.collection("finance_contributions")
                  .where("made_to",    "==", cand["candidate_name"])
                  .where("cycle_key",  "==", cycle_key)
        )

        donors: Dict[str, Dict[str, float|int|str]] = {}
        for d in q.stream():
            row = d.to_dict()
            name = row["contributor"]
            agg  = donors.setdefault(name, {
                "name" : name,
                "total": 0.0,
                "txns" : 0,
                "type" : row.get("contributor_type", "individual")
            })
            agg["total"] += float(row["amount"])
            agg["txns"]  += 1

        # Split + sort
        individual = sorted(
            (v for v in donors.values() if v["type"] != "pac/org"),
            key=lambda x: x["total"], reverse=True)[:limit]
        pac = sorted(
            (v for v in donors.values() if v["type"] == "pac/org"),
            key=lambda x: x["total"], reverse=True)[:limit]

        return {
            "individual": individual,
            "pac": pac,
            "cycle": cycle_key
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Top donors fetch failed")
        raise HTTPException(500, f"Failed to fetch top donors: {exc}") from exc


# ───────────────── donor ▸ raw contributions (paginated) ──────────────── #
@router.get("/donors/{name}/contributions")
async def donor_contributions(
    name: str,
    cycle: Optional[str] = Query(None),
    page_size: int       = Query(100, ge=1, le=MAX_PAGE),
    page_token: Optional[str] = Query(None),
    client                    = Depends(get_firestore_client),
):
    """
    Paginated stream of all contributions by a donor (optionally filtered by cycle).
    Cursor = base64(receipt_date + doc_id).
    """
    try:
        q: FsQuery = (
            client.collection("finance_contributions")
                  .where("contributor", "==", name)
        )
        if cycle:
            q = q.where("cycle_key", "==", cycle)

        q = (q.order_by("receipt_date", direction=firestore.Query.DESCENDING)
               .order_by("__name__",      direction=firestore.Query.DESCENDING))

        if page_token:
            ts, doc_id = decode_cursor(page_token)
            q = q.start_after({"receipt_date": ts, "__name__": doc_id})

        docs = list(q.limit(page_size).stream())
        rows = [d.to_dict() for d in docs]

        next_token = (
            encode_cursor(rows[-1]["receipt_date"], docs[-1].id)
            if len(rows) == page_size else None
        )

        return {
            "rows": rows,
            "next_page_token": next_token,
            "page_size": page_size,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Donor rows fetch failed")
        raise HTTPException(500, f"Failed to fetch contributions: {exc}") from exc
