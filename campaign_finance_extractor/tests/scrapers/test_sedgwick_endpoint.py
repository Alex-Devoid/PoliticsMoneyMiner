# tests/scrapers/test_sedgwick_endpoint.py
"""
Standalone integration-test for the Sedgwick scraping endpoint.

We *do not* import cloud_function.main because that triggers Settings()
validation against prod-only env-vars.  Instead we assemble a minimal FastAPI
app that mounts just the /sedgwick router.
"""
import os, sys, time, threading, requests, uvicorn

# ---------------------------------------------------------------------------
# local imports – ensure repo root is on the path
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(REPO_ROOT)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cloud_function.routes.sedgwick import router as sedgwick_router
from cloud_function.scrapers.SedgwickElectionsCampaignExpenseReport import (
    SedgwickExpenseScraper,
)

_schema_probe = SedgwickExpenseScraper(ocr=False, use_openai=False)
FilingModel = _schema_probe.Filing

# ---------------------------------------------------------------------------
# minimal app ⇢ only the route we care about
# ---------------------------------------------------------------------------
app = FastAPI()
app.include_router(sedgwick_router)
client = TestClient(app)


def _run_server() -> None:
    """Launch the minimal FastAPI app in a background Uvicorn worker."""
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="warning")


# ---------------------------------------------------------------------------
# actual test
# ---------------------------------------------------------------------------
def test_scrape_endpoint() -> None:
    # start server once per test-process
    svr = threading.Thread(target=_run_server, daemon=True)
    svr.start()
    time.sleep(3)  # give Uvicorn time to bind

    resp = requests.post(
        "http://0.0.0.0:8081/sedgwick/scrape",
        params=dict(
            year=2023,
            office="WICHITA CITY MAYOR",
            ocr=True,
            save_pdfs=True,
            limit=3,
        ),
        timeout=600_000,
    )

    # ---------------- basic assertions ----------------
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert payload["count"] == 3
    assert len(payload["pages"]) == 3
    assert len(payload["filings"]) == 3

    # legacy page-stub still present
    first_page = payload["pages"][0]
    assert {"page", "pageImg", "totalPages", "rows"} <= first_page.keys()
    assert first_page["rows"] and "bbox" in first_page["rows"][0]

    # ---------------- schema validation ----------------
    for wrapper in payload["filings"]:
        validated = FilingModel.model_validate(wrapper["filing"])
        # quick smoke-check
        assert validated.report_meta.candidate_name.value
