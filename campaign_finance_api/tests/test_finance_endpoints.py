# tests/test_finance_endpoints.py
"""
End-to-end tests for the /finance routes.

• spins up the FastAPI app (Uvicorn in a background thread)
• seeds the Firestore emulator with a tiny fixture set
• calls every public endpoint and prints live responses
"""
import os, sys, time, threading
from datetime import datetime, timedelta, timezone

import pytest, requests, uvicorn

# ───────────── env + imports (set vars before importing app) ────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "firestore-emulator:8080")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dev-emulated-project")
os.environ.setdefault("FIRESTORE_DATABASE_ID", "(default)")

from cloud_function.main                  import app
from cloud_function.utils.firestore_utils import get_firestore_client

HOST, PORT = "0.0.0.0", 8082
BASE       = f"http://{HOST}:{PORT}/finance"

CAND_SLUG = "rachel-roarx"
CAND_NAME = "Rachel Roarx"
CYCLE_KEY = "2024_GENERAL"
IND_NAME  = "John Smith"
PAC_NAME  = "ACME PAC"
STATE_ABB = "KY"

# ────────────────────────── server fixture ──────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def _fastapi_server():
    t = threading.Thread(
        target=uvicorn.run,
        kwargs=dict(app=app, host=HOST, port=PORT, log_level="error"),
        daemon=True,
    )
    t.start()
    time.sleep(3)
    yield

# ─────────────────────── Firestore fixture ──────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def _firestore_fixture():
    client = get_firestore_client()

    # ── summary document ──
    client.collection("finance_summary").document(CAND_SLUG).set(
        {
            "candidate_slug": CAND_SLUG,
            "candidate_name": CAND_NAME,
            "office": "STATE REPRESENTATIVE",
            "district": "38TH DISTRICT",
            "jurisdiction": STATE_ABB,
            "cycles": {
                CYCLE_KEY: {
                    "raised_total": 400.0,
                    "individual_total": 100.0,
                    "individual_count": 1,
                    "pac_total": 300.0,
                    "last_updated": datetime.utcnow().replace(tzinfo=timezone.utc),
                }
            },
        }
    )

    # ── contribution rows ──
    contrib_col = client.collection("finance_contributions")
    meta_c = {
        "office": "STATE REPRESENTATIVE",
        "district": "38TH DISTRICT",
        "jurisdiction": STATE_ABB,
        "filing_type": "contribution",
        "election_date": "11/05/2024",
        "election_type": "GENERAL",
        "cycle_key": CYCLE_KEY,
        "made_to": CAND_NAME,
        "candidate_slug": CAND_SLUG,
    }

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    batch = client.batch()
    # individual
    batch.set(
        contrib_col.document(),
        {
            **meta_c,
            "contributor": IND_NAME,
            "contributor_lc": IND_NAME.lower(),
            "contributor_type": "individual",
            "amount": 100.0,
            "receipt_date": now - timedelta(days=1),
        },
    )
    # three PAC rows
    for off, amt in enumerate((150.0, 100.0, 50.0), start=2):
        batch.set(
            contrib_col.document(),
            {
                **meta_c,
                "contributor": PAC_NAME,
                "contributor_lc": PAC_NAME.lower(),
                "contributor_type": "pac/org",
                "amount": amt,
                "receipt_date": now - timedelta(days=off),
            },
        )
    batch.commit()

    # ── expenditure rows ──
    expend_col = client.collection("finance_expenditures")
    meta_e = {
        "office": "STATE REPRESENTATIVE",
        "district": "38TH DISTRICT",
        "jurisdiction": STATE_ABB,
        "filing_type": "expenditure",
        "election_date": "11/05/2024",
        "election_type": "GENERAL",
        "cycle_key": CYCLE_KEY,
        "candidate_slug": CAND_SLUG,
    }

    expend_col.add(
        {
            **meta_e,
            "payee": "Widgets Inc.",
            "amount": 20.0,
            "expenditure_date": now - timedelta(days=1),
        }
    )
    expend_col.add(
        {
            **meta_e,
            "payee": "Postmaster",
            "amount": 30.0,
            "expenditure_date": now - timedelta(days=2),
        }
    )

    yield

    # ── cleanup (best-effort) ──
    try:
        client.collection("finance_summary").document(CAND_SLUG).delete()
        for d in contrib_col.where("candidate_slug", "==", CAND_SLUG).stream():
            d.reference.delete()
        for d in expend_col.where("candidate_slug", "==", CAND_SLUG).stream():
            d.reference.delete()
    except Exception:
        # If the emulator shuts down first we don’t want the whole suite to error
        pass

# ───────────────────────────── tests ─────────────────────────────────────
def test_states():
    """GET /finance/states returns the distinct jurisdiction list."""
    r = requests.get(f"{BASE}/states")
    print("\nfinance_states →", r.status_code, r.json())
    assert r.status_code == 200
    body = r.json()
    states = body["states"] if isinstance(body, dict) and "states" in body else body
    assert STATE_ABB in states

def test_lookup_candidates():
    r = requests.get(f"{BASE}/lookup/candidates", params={"q": "roa"})
    print("\nlookup_candidates →", r.status_code, r.json())
    assert r.status_code == 200
    assert any(h["slug"] == CAND_SLUG for h in r.json())

def test_lookup_donors():
    # narrow prefix so our seeded row is within the first-20 window
    r1 = requests.get(f"{BASE}/lookup/donors", params={"q": "acm"})
    print("\nlookup_donors (acm) →", r1.status_code, r1.json())
    assert r1.status_code == 200
    assert any(h["name"] == PAC_NAME for h in r1.json())

    r2 = requests.get(f"{BASE}/lookup/donors", params={"q": "john sm"})
    print("lookup_donors (john sm) →", r2.status_code, r2.json())
    assert r2.status_code == 200
    assert any(h["name"] == IND_NAME for h in r2.json())

def test_candidate_summary():
    r = requests.get(
        f"{BASE}/candidates/{CAND_SLUG}/summary", params={"cycle": CYCLE_KEY}
    )
    print("\ncandidate_summary →", r.status_code, r.json())
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["raised_total"] == pytest.approx(400.0)

def test_candidate_spending():
    r = requests.get(
        f"{BASE}/candidates/{CAND_SLUG}/spending", params={"cycle": CYCLE_KEY}
    )
    print("\ncandidate_spending →", r.status_code, r.json())
    body = r.json()
    assert r.status_code == 200
    assert body["spent_total"] == pytest.approx(50.0)
    assert body["txns"] == 2

def test_top_donors():
    r = requests.get(
        f"{BASE}/candidates/{CAND_SLUG}/top-donors",
        params={"cycle": CYCLE_KEY, "limit": 100},
    )
    print("\ntop_donors →", r.status_code, r.json())
    body = r.json()
    # our ACME PAC entry should be present with a 300-total,
    # even if bigger PACs exist from external seed data
    assert any(
        p["name"] == PAC_NAME and p["total"] == pytest.approx(300.0)
        for p in body["pac"]
    )

def test_raised_since():
    since = (
        datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=1)
    ).date().isoformat()
    r = requests.get(
        f"{BASE}/candidates/{CAND_SLUG}/raised-since", params={"after": since}
    )
    print("\nraised_since →", r.status_code, r.json())
    body = r.json()
    assert r.status_code == 200
    # only the 100-dollar individual gift is within ≤1 day
    assert body["raised_total"] == pytest.approx(100.0)

def test_donor_contributions_paging():
    first = requests.get(
        f"{BASE}/donors/{PAC_NAME}/contributions",
        params={"cycle": CYCLE_KEY, "page_size": 2},
    )
    print("\ndonor_contributions page-1 →", first.status_code, first.json())
    first_j = first.json()
    assert len(first_j["rows"]) == 2

    second = requests.get(
        f"{BASE}/donors/{PAC_NAME}/contributions",
        params={
            "cycle": CYCLE_KEY,
            "page_size": 2,
            "page_token": first_j["next_page_token"],
        },
    )
    print("donor_contributions page-2 →", second.status_code, second.json())
    assert len(second.json()["rows"]) == 1
