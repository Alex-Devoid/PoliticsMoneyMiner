"""
tests/routes/test_kentucky.py
─────────────────────────────
Full-stack integration test for the /kentucky/finance endpoint.

• The FastAPI router runs exactly as in production (background task included)
• The scraper executes _stream_csv() and _commit() against the Firestore
  emulator — we verify the written docs.
• External HTTP requests to the KREF portal are stubbed with a tiny CSV
  fixture so the test is hermetic and fast.
"""
from __future__ import annotations
import os, sys, time, threading, requests, uvicorn
from pathlib import Path

# ---------------------------------------------------------------------------
# local imports – ensure repo root is on the path
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(REPO_ROOT)

import os
import time
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.cloud import firestore

# ---------------------------------------------------------------------------
#  Ensure we point at the running Firestore emulator
# ---------------------------------------------------------------------------
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:8080")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dev-emulated-project")
os.environ.setdefault("FIRESTORE_DATABASE_ID", "(default)")

# ---------------------------------------------------------------------------
#  import AFTER env-vars so the client picks them up
# ---------------------------------------------------------------------------
from cloud_function.routes import kentucky as ky_router
from cloud_function.utils.firestore_utils import get_firestore_client

# ---------------------------------------------------------------------------
#  minimal app – only the KY router
# ---------------------------------------------------------------------------
app = FastAPI()
app.include_router(ky_router.router)
client = TestClient(app)

# ---------------------------------------------------------------------------
#  CSV fixture served in place of the real portal export
# ---------------------------------------------------------------------------
_FAKE_CSV = (
    "Amount,Receipt Date,Recipient First Name,Recipient Last Name,"
    "Contributor First Name,Contributor Last Name,Office Sought,Location,"
    "Contribution Type\n"
    "100.00,1/1/2025,Jane,Doe,John,Smith,STATE REPRESENTATIVE,38TH DISTRICT,"
    "Individual\n"
    "250.00,1/2/2025,Jane,Doe,ACME,,STATE REPRESENTATIVE,38TH DISTRICT,"
    "PAC\n"
).encode()


class _StubResp:
    status_code = 200
    content = _FAKE_CSV

    def raise_for_status(self) -> None:  # noqa: D401
        return


# ---------------------------------------------------------------------------
#  the actual test
# ---------------------------------------------------------------------------
@mock.patch("cloud_function.scrapers.KentuckyFinanceScraper.requests.Session.get")
def test_kentucky_finance_full_stack(mock_get) -> None:
    """
    POST /kentucky/finance (mode=contrib) and verify Firestore side-effects.
    """
    # stub network
    mock_get.return_value = _StubResp()

    # wipe emulator collections first
    db = get_firestore_client()
    for col in ("finance_contributions", "finance_summary"):
        for doc in db.collection(col).stream():
            doc.reference.delete()

    # ------------------------------------------------------------------
    # fire the request (TestClient waits for background tasks to finish)
    # ------------------------------------------------------------------
    resp = client.post(
        "/kentucky/finance",
        json={
            "election_date": "11/5/2024",
            "election_type": "GENERAL",
            "office": "STATE REPRESENTATIVE",
            "mode": "contrib",
        },
        timeout=30,
    )
    assert resp.status_code == 202, resp.text

    # ------------------------------------------------------------------
    # verify Firestore writes
    # ------------------------------------------------------------------
    contrib_rows = list(db.collection("finance_contributions").stream())
    assert len(contrib_rows) == 2, "expected two raw contribution docs"

    # summary doc keyed by candidate-slug “jane-doe”
    summary_ref = db.collection("finance_summary").document("jane-doe")
    summary_doc = summary_ref.get()
    assert summary_doc.exists, "summary document missing"

    cycles = summary_doc.to_dict()["cycles"]
    assert "2024_GENERAL" in cycles
    cycle_stats = cycles["2024_GENERAL"]
    assert cycle_stats["raised_total"] == 350.00
    assert cycle_stats["individual_total"] == 100.00
    assert cycle_stats["individual_count"] == 1
    assert cycle_stats["pac_total"] == 250.00
