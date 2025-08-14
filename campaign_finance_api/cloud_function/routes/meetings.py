# cloud_function/routes/meetings.py
from fastapi import APIRouter, Query, HTTPException, Depends
from typing     import Optional, List, Tuple
from datetime   import datetime
from itertools  import product
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter, Query as FsQuery
from google.api_core.exceptions import FailedPrecondition
from cloud_function.utils.firestore_utils import get_firestore_client
import base64, json, logging, asyncio

router  = APIRouter()
MAX_IN  = 10                                 # Firestore IN-filter cap
MAX_FAN = 50                                 # sanity-cap for sub-queries

# ───────────────────────── helpers ──────────────────────────
def split(q: Optional[str]) -> List[str]:
    return [s.strip() for s in q.split(",") if s.strip()] if q else []

def encode_cursor(last_doc: dict) -> str:
    blob = json.dumps(
        [last_doc["meeting_date"], last_doc["meeting_id"]], default=str
    ).encode()
    return base64.urlsafe_b64encode(blob).decode()

def decode_cursor(token: str) -> Tuple[datetime, str]:
    ts_str, mid = json.loads(base64.urlsafe_b64decode(token.encode()))
    return datetime.fromisoformat(ts_str), mid

def get_total_count(q: FsQuery) -> int:
    """Fast server-side COUNT with silent fallback to client-side stream."""
    try:
        agg = q.count().get()
        return (agg[0] or {}).get("count", 0)
    except Exception:
        return sum(1 for _ in q.stream())

def build_query(
    client,
    state: Optional[List[str]] = None,
    place: Optional[List[str]] = None,
    committee: Optional[List[str]] = None,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
) -> FsQuery:
    q: FsQuery = client.collection("scraped_meetings")
    if state:     q = q.where(filter=FieldFilter("state", "in", state))
    if place:     q = q.where(filter=FieldFilter("place", "in", place))
    if committee:
        q = (q.where(filter=FieldFilter("committee_name", "==", committee[0]))
             if len(committee) == 1 else
             q.where(filter=FieldFilter("committee_name", "in", committee)))
    if start_ts:  q = q.where(filter=FieldFilter("meeting_date", ">=", start_ts))
    if end_ts:    q = q.where(filter=FieldFilter("meeting_date", "<=", end_ts))

    return (q.order_by("meeting_date", direction=firestore.Query.DESCENDING)
             .order_by("meeting_id",   direction=firestore.Query.DESCENDING))

# ───────────────────────── slice endpoint ───────────────────
@router.get("/")
async def get_meetings(
    state:      Optional[str] = Query(None),
    place:      Optional[str] = Query(None),
    committee:  Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
    page_size:  int           = Query(30, ge=1, le=200),
    page_token: Optional[str] = Query(None),
    client                      = Depends(get_firestore_client),
):
    try:
        states     = split(state)[:MAX_IN]
        places     = split(place)[:MAX_IN]
        committees = split(committee)[:MAX_IN]
        start_ts   = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_ts     = datetime.strptime(end_date,   "%Y-%m-%d") if end_date   else None

        q = build_query(client, states, places, committees, start_ts, end_ts)

        if page_token:
            ts, mid = decode_cursor(page_token)
            q = q.start_after({"meeting_date": ts, "meeting_id": mid})

        docs       = [d.to_dict() for d in q.limit(page_size).stream()]
        next_token = encode_cursor(docs[-1]) if len(docs) == page_size else None

        return {
            "page_size"      : page_size,
            "data"           : docs,
            "next_page_token": next_token,
            "total"          : None,      # fetched lazily
        }

    except Exception as exc:
        logging.exception("Meeting query failed")
        raise HTTPException(500, f"Failed to fetch meetings: {exc}") from exc

# ───────────────────────── count endpoint ───────────────────
@router.get("/count")
async def get_meetings_count(
    state:      Optional[str] = Query(None),
    place:      Optional[str] = Query(None),
    committee:  Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
    client                      = Depends(get_firestore_client),
):
    try:
        states     = split(state)[:MAX_IN]
        places     = split(place)[:MAX_IN]
        committees = split(committee)[:MAX_IN]
        start_ts   = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_ts     = datetime.strptime(end_date,   "%Y-%m-%d") if end_date   else None

        # Fast path – zero or one IN filter
        in_filters = sum(bool(x and len(x) > 1) for x in (states, places, committees))
        if in_filters <= 1:
            q = build_query(client, states, places, committees, start_ts, end_ts)
            return {"total": get_total_count(q)}

        # Fan-out path – more than one IN filter
        if len(states) * len(places or [None]) * len(committees or [None]) > MAX_FAN:
            raise HTTPException(400, "Query fan-out too large")

        # Build every valid Cartesian combination, replacing extra IN lists
        # with single-value equality filters.
        async def sub_count(st, pl, cm):
            q = build_query(
                client,
                [st] if st else None,
                [pl] if pl else None,
                [cm] if cm else None,
                start_ts,
                end_ts,
            )
            return get_total_count(q)

        combos = product(states or [None],
                         places  or [None],
                         committees or [None])

        totals = await asyncio.gather(*(sub_count(*c) for c in combos))
        return {"total": sum(totals)}

    except FailedPrecondition as exc:
        # fall back (very rare once indexes are created)
        logging.warning("Firestore failed‐precondition in count: %s", exc)
        return {"total": 0}
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Meeting count failed")
        raise HTTPException(500, f"Failed to count meetings: {exc}") from exc
