# tests/test_qc_endpoints.py  – drop‑in
"""
End‑to‑end smoke‑test for the three QC routes the Nuxt dashboard calls.

• Spins up FastAPI in‑process (Uvicorn thread)
• Seeds *both* Firestore‑ and Storage‑emulators with dummy data
• Confirms the round‑trip approve‑row behaviour
"""
from __future__ import annotations

import base64, io, os, socket, sys, threading, time, uuid
from datetime import timedelta
from typing import Tuple

import pytest, requests, uvicorn
from google.cloud import firestore, storage
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from PIL import Image                                            # pillow in requirements.txt

# ──────────── env BEFORE importing FastAPI app ────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "firestore-emulator:8080")
os.environ.setdefault("STORAGE_EMULATOR_HOST" , "firebase-emulator:9199")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT",     "dev-emulated-project")
os.environ.setdefault("FIRESTORE_DATABASE_ID",    "(default)")

from cloud_function.main                  import app                    # noqa: E402
from cloud_function.utils.firestore_utils import get_firestore_client   # noqa: E402
from cloud_function.utils.storage_utils   import get_storage_client     # noqa: E402, we’ll re‑use helpers

# ──────────── seed helper ──────────────────────────────────────────────
def _seed_fixtures() -> Tuple[str, str, str]:
    """
    Writes one candidate, one filing, one row and one PNG into the emulators.
    Returns (slug, doc_id, row_id).
    """
    db   = get_firestore_client()
    gcs  = get_storage_client()

    slug     = "test-candidate"
    doc_id   = uuid.uuid4().hex
    row_id   = uuid.uuid4().hex
    cycle    = "2024_GENERAL"
    bucket   = "sedgwick-finance-files"
    blob_key = f"{slug}/{doc_id}/page_1.png"
    gs_uri   = f"gs://{bucket}/{blob_key}"

    # -- tiny 1×1 PNG ----------------------------------------------------
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color="white").save(buf, format="PNG")
    gcs.bucket(bucket).blob(blob_key).upload_from_string(buf.getvalue(),
                                                         content_type="image/png")

    # -- Firestore docs --------------------------------------------------
    db.collection("finance_summary").document(slug).set({
        "candidate_name": "Test Candidate",
        "office"        : "Testing Office",
        "jurisdiction"  : "SEDGWICK_KS",
        "last_filed_doc": doc_id,
        "cycles"        : {cycle: {}},
        "updated_at"    : SERVER_TIMESTAMP,
    })

    db.collection("finance_files").document(f"{slug}__{doc_id}").set({
        "office" : "Testing Office",
        "cycle"  : cycle,
        "files"  : {"page_1": gs_uri, "pdf": "gs://bucket/dummy.pdf"},
        "uploaded": SERVER_TIMESTAMP,
    })

    db.collection("finance_contributions").document(row_id).set({
        "candidate_slug": slug,
        "cycle_key"    : cycle,
        "doc_id"       : doc_id,
        "row_type"     : "contribution",
        "contributor"  : "Dummy Donor",
        "amount"       : {"value": 42.0},
        "_page"        : 1,
        "validated"    : False,
    })
    return slug, doc_id, row_id

SLUG, DOC_ID, ROW_ID = _seed_fixtures()

# ──────────── start FastAPI on a free port ─────────────────────────────
def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

PORT = _free_port()
BASE = f"http://127.0.0.1:{PORT}"

@pytest.fixture(scope="session", autouse=True)
def _run_server():
    th = threading.Thread(
        target=uvicorn.run,
        kwargs=dict(app=app, host="127.0.0.1", port=PORT, log_level="warning"),
        daemon=True,
    )
    th.start()
    for _ in range(20):
        try:
            requests.get(f"{BASE}/", timeout=0.5); break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    else:
        pytest.skip("FastAPI failed to start")
    yield

# ═════════════════════ tests ════════════════════════════════════════════
def test_overview_and_page_and_qc():
    # 1️⃣ overview --------------------------------------------------------
    ov = requests.get(f"{BASE}/finance/filings", timeout=5).json()
    assert any(d["id"] == SLUG for d in ov)

    # 2️⃣ page‑payload ----------------------------------------------------
    pg = requests.get(f"{BASE}/finance/filings/{SLUG}/pages",
                      params={"page": 1}, timeout=5).json()
    assert pg["pageImg"].startswith("http") and pg["rows"]

    # 3️⃣ approve‑row -----------------------------------------------------
    body = {"doc_id": DOC_ID, "page": 1, "row_id": ROW_ID, "approved": True}
    r = requests.post(f"{BASE}/finance/qc/approve-row", json=body, timeout=5)
    assert r.status_code == 204

    snap = (get_firestore_client()
            .collection("finance_contributions").document(ROW_ID).get())
    assert snap.exists and snap.to_dict()["validated"] is True
