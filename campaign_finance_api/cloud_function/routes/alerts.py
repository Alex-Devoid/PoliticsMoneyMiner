from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Tuple
from google.cloud import firestore
import datetime, logging
from urllib.parse import quote

from cloud_function.utils.firestore_utils import get_firestore_client
from cloud_function.utils.email_service   import send_to_microservice

logger  = logging.getLogger(__name__)
router  = APIRouter()


# --------------------------------------------------------------------------- #
#  Data models
# --------------------------------------------------------------------------- #
class HierarchicalBody(BaseModel):
    email: str
    state: str
    place: str
    committees: List[str]


# --------------------------------------------------------------------------- #
#   Helper utilities
# --------------------------------------------------------------------------- #
def to_valid_doc_id(s: str) -> str:
    """Percent‑encode anything Firestore would treat as a path separator."""
    return quote(s, safe='')


def make_naive(dt: datetime.datetime | None) -> datetime.datetime | None:
    """Convert offset‑aware → naive; pass through None/naive."""
    if dt and dt.tzinfo is not None:
        return dt.astimezone(tz=None).replace(tzinfo=None)
    return dt


# --------------------------------------------------------------------------- #
#  Lookup endpoints (states / places / committees)
# --------------------------------------------------------------------------- #
@router.get("/states")
async def get_states(client=Depends(get_firestore_client)):
    """
    Return a DISTINCT list of states present in `scraped_meetings`.
    """
    try:
        states = {
            d.to_dict().get("state")
            for d in client.collection("scraped_meetings").stream()
            if d.to_dict().get("state")
        }
        return {"states": sorted(states)}
    except Exception as e:
        logger.exception("Error fetching states")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/places")
async def get_places(state: str, client=Depends(get_firestore_client)):
    """
    Return DISTINCT places for a given state from `scraped_meetings`.
    """
    try:
        places = {
            d.to_dict().get("place")
            for d in (
                client.collection("scraped_meetings")
                      .where("state", "==", state)
                      .stream()
            )
            if d.to_dict().get("place")
        }
        return {"places": sorted(places)}
    except Exception as e:
        logger.exception("Error fetching places")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/committees")
async def get_committees(state: str, place: str, client=Depends(get_firestore_client)):
    """
    Return DISTINCT committees for a given state + place.
    """
    try:
        committees = {
            d.to_dict().get("committee_name")
            for d in (
                client.collection("scraped_meetings")
                      .where("state",  "==", state)
                      .where("place",  "==", place)
                      .stream()
            )
            if d.to_dict().get("committee_name")
        }
        return {"committees": sorted(committees)}
    except Exception as e:
        logger.exception("Error fetching committees")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# ➕  Subscribe (hierarchical doc structure)
# --------------------------------------------------------------------------- #
@router.post("/subscribe")
async def subscribe_hierarchical(body: HierarchicalBody,
                                 client=Depends(get_firestore_client)):
    try:
        email = body.email.strip().lower()
        if not email:
            raise HTTPException(400, "Email is required.")
        if not body.committees:
            raise HTTPException(400, "Must provide at least one committee.")

        doc_ref = client.collection("alerts_subscriptions").document(email)
        doc     = doc_ref.get().to_dict() or {
            "email": email,
            "created_at": datetime.datetime.utcnow(),
            "states": {}
        }

        states   = doc.setdefault("states", {})
        places   = states.setdefault(body.state, {}).setdefault("places", {})
        committees = places.setdefault(body.place, {}).setdefault("committees", {})

        for c in body.committees:
            committees.setdefault(c, {"last_sent_at": None})

        doc_ref.set(doc)
        return {"detail": "Subscription(s) updated successfully."}

    except Exception as e:
        logger.exception("Error creating/updating subscription")
        raise HTTPException(500, detail=str(e))


# --------------------------------------------------------------------------- #
#    Send alerts (one e‑mail per subscriber)
# --------------------------------------------------------------------------- #
@router.post("/send")
async def send_alerts_hierarchical(client=Depends(get_firestore_client)):
    """
    • One email per subscriber summarising *new* docs.  
    • Sentinel doc (`alerts_sent/{email}_{meeting}_{asset}`) ensures idempotency.  
    • `last_sent_at` bumped only for committees actually alerted.  
    """

    now      = datetime.datetime.utcnow()
    subs     = list(client.collection("alerts_subscriptions").stream())
    scraped  = {
        d.id: d.to_dict()
        for d in client.collection("scraped_meetings")
                       .where("documents", "!=", None)
                       .stream()
    }

    logger.info("Alert run started – %d subscribers, %d scraped meetings",
                len(subs), len(scraped))

    for sub_doc in subs:
        sub   = sub_doc.to_dict()
        email = sub.get("email")
        if not email:
            continue

        new_docs: List[Dict]                       = []
        sentinels: List[firestore.DocumentReference] = []
        bumps:     set[Tuple[str, str, str]]       = set()

        # walk the hierarchy ------------------------------------------------
        for st, st_val in sub.get("states", {}).items():
            for pl, pl_val in st_val.get("places", {}).items():
                for cm, cm_val in pl_val.get("committees", {}).items():
                    last_sent = make_naive(cm_val.get("last_sent_at")) \
                                or datetime.datetime(1970, 1, 1)

                    for m_id, m in scraped.items():
                        if (m["state"], m["place"], m["committee_name"]) != (st, pl, cm):
                            continue

                        for asset in m.get("documents", []):
                            sent_id  = f"{sub_doc.id}_{m_id}_{to_valid_doc_id(asset['asset_url'])}"
                            sent_ref = client.collection("alerts_sent").document(sent_id)
                            if sent_ref.get().exists:
                                continue

                            if make_naive(asset["scrape_date"]) <= last_sent:
                                continue

                            new_docs.append({
                                "committee": cm,
                                "state": st,
                                "place": pl,
                                "asset_type": asset["asset_type"],
                                "meeting_name": m["asset_name"],
                                "meeting_date": m["meeting_date"],
                                "url": asset["asset_url"]
                            })
                            sentinels.append(sent_ref)
                            bumps.add((st, pl, cm))

        if not new_docs:
            continue

        # build + send e‑mail ----------------------------------------------
        subject = f"New Documents in Your Subscriptions ({len(new_docs)})"
        body    = ["Hello,<br/>",
                   "Here are the new or updated documents in your local government subscriptions:<br/><ul>"]
        for d in new_docs:
            body.append(
                f"<li><strong>{d['asset_type'].title()}</strong> – "
                f"{d['committee']} in {d['place'].title()}, {d['state'].upper()}<br/>"
                f"Meeting: {d['meeting_name']}<br/>"
                f"Date: {d['meeting_date']}<br/>"
                f"<a href='{d['url']}' target='_blank'>View document</a></li><br/>")
        body.append("</ul><br/>Regards,<br/>Your Local Gov Alerts")

        send_to_microservice({
            "recipients": [email],
            "subject": subject,
            "body": "\n".join(body),
            "csv_data": [],
            "send_individual": False,
            "use_batch": False
        })
        logger.info("Alert sent to %s with %d doc(s)", email, len(new_docs))

        # commit sentinels + bump timestamps -------------------------------
        batch = client.batch()
        for ref in sentinels:
            batch.set(ref, {"sent_at": now})
        batch.commit()

        for st, pl, cm in bumps:
            sub["states"][st]["places"][pl]["committees"][cm]["last_sent_at"] = now
        client.collection("alerts_subscriptions").document(sub_doc.id).set(sub, merge=True)

    return {"detail": "Alert run complete"}
