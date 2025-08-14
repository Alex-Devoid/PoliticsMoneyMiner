"""
tests/scrapers/test_kentucky_live.py
────────────────────────────────────
Full‑stack integration test for the /kentucky/finance endpoint that exercises
Firestore **and** confirms a CSV file lands on disk.

**Change**: the CSV now persists in a deterministic folder, *next to this
test file*, instead of a throw‑away tmpdir. That means every run appends to the
same location, making it easy to inspect artefacts after CI or local runs.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Final

from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── repo root on the path ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# ----------------------------------------------------------------------
#  Environment – Firestore emulator + CSV dir
# ----------------------------------------------------------------------
# Allow callers to pre‑set FIRESTORE_EMULATOR_HOST but fall back if absent.
os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "firestore-emulator:8080")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dev-emulated-project")
os.environ.setdefault("FIRESTORE_DATABASE_ID", "(default)")

# Persist CSVs in a stable folder under the test directory
CSV_OUT_DIR: Final = Path(__file__).resolve().parent / "ky_csv_out"
CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["KYFIN_CSV_DIR"] = str(CSV_OUT_DIR)

print(
    "Environment Variables:",
    *(f"\n  {k} = {v}" for k, v in (
        ("FIRESTORE_EMULATOR_HOST", os.environ["FIRESTORE_EMULATOR_HOST"]),
        ("GOOGLE_CLOUD_PROJECT"   , os.environ["GOOGLE_CLOUD_PROJECT"]),
        ("FIRESTORE_DATABASE_ID"  , os.environ["FIRESTORE_DATABASE_ID"]),
        ("KYFIN_CSV_DIR"          , os.environ["KYFIN_CSV_DIR"]),
    )),
)

# ── AFTER env‑vars so the client inherits them -------------------------
from google.api_core.exceptions import ServiceUnavailable  # noqa: E402
from google.cloud import firestore  # noqa: E402
from cloud_function.routes import kentucky as ky_router  # noqa: E402
from cloud_function.utils.firestore_utils import get_firestore_client  # noqa: E402

# ----------------------------------------------------------------------
#  Mini FastAPI app with only the /kentucky routes
# ----------------------------------------------------------------------
app = FastAPI()
app.include_router(ky_router.router)
client = TestClient(app)

ELECT_DATE     = "11/5/2024"
ELECT_TYPE     = "GENERAL"
DEFAULT_OFFICE = "STATE SENATOR (ODD)"
CYCLE_KEY      = f"{datetime.strptime(ELECT_DATE, '%m/%d/%Y').year}_{ELECT_TYPE}"

# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

def _wipe(col_name: str, db: firestore.Client) -> None:
    """Delete every document in *col_name* inside the emulator. Skip if unreachable."""
    import pytest
    try:
        for doc in db.collection(col_name).stream():
            doc.reference.delete()
    except ServiceUnavailable as exc:
        pytest.skip(f"Firestore emulator unreachable ({exc}); skipping live test.")


def _wait_for_rows(db: firestore.Client, col: str, timeout: int = 120) -> int:
    """Poll until at least one document appears or timeout elapses."""
    import pytest
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            rows = list(db.collection(col).stream())
        except ServiceUnavailable as exc:
            pytest.skip(f"Firestore emulator unreachable ({exc}); skipping live test.")
        if rows:
            return len(rows)
        time.sleep(3)
    return 0


# ══════════════════════════════════════════════════════════════════════
#  TEST
# ══════════════════════════════════════════════════════════════════════

def test_kentucky_finance_live() -> None:
    db = get_firestore_client()
    print("\nFirestore project:", db.project)



    # 2️⃣ trigger the scrape
    print("\n🚀  POST /kentucky/finance …")
    t0 = datetime.now()
    resp = client.post(
        "/kentucky/finance",
        json={
            "election_date": ELECT_DATE,
            "election_type": ELECT_TYPE,
            "office"       : DEFAULT_OFFICE,
            "mode"         : "contrib",
        },
    )
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  ↳ {resp.status_code} in {elapsed:.1f}s – {resp.json()}")
    assert resp.status_code == 202, resp.text

    import pytest
    if resp.json().get("rows_enqueued", 0) == 0:
        pytest.skip("KREF returned 0 rows; skipping live test.")

    # 3️⃣ wait for Firestore writes
    count = _wait_for_rows(db, "finance_contributions")
    assert count, "No contribution docs appeared within the timeout window"
    print(f"📥  {count} raw rows in Firestore")

    # 4️⃣ verify summary roll‑up for THIS cycle
    summaries = list(db.collection("finance_summary").stream())
    matches   = [s for s in summaries if CYCLE_KEY in s.to_dict().get("cycles", {})]
    assert matches, f"No summary doc contained cycle '{CYCLE_KEY}'"
    cyc_data = matches[0].to_dict()["cycles"][CYCLE_KEY]
    assert cyc_data["raised_total"] > 0, "raised_total not populated"
    print("📊  Summary roll‑up ok → raised_total =", cyc_data["raised_total"])

    # 5️⃣ CSV file exists and has ≥1 data row
    csv_files = list(CSV_OUT_DIR.glob("ky_contributions_*.csv"))
    assert csv_files, "CSV file was not generated"
    with csv_files[0].open() as fp:
        header = next(fp)
        first_row = next(fp, None)
    assert first_row, "CSV contains header only"
    print("📄  CSV persisted →", csv_files[0])

    print("\n✅  LIVE Kentucky scrape test succeeded!\n")
