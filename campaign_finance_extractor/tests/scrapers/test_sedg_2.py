# tests/scrapers/test_sedgwick_endpoint.py
"""
Integration test that processes **one local PDF** through the /sedgwick/scrape
endpoint (which, in local-file mode, never touches the remote portal).
"""
import os, sys, time, threading, requests, uvicorn
from pathlib import Path

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

# absolute path to the single test PDF
PDF_PATH = (
    Path(__file__)
    .parent                                   # tests/scrapers
    .joinpath("test_example_page_5.pdf")
    .resolve()
)

_schema_probe = SedgwickExpenseScraper(ocr=False, use_openai=False)
FilingModel   = _schema_probe.Filing

# ---------------------------------------------------------------------------
# minimal app ⇢ only the route we care about
# ---------------------------------------------------------------------------
app    = FastAPI()
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
            local_pdf=str(PDF_PATH),
            ocr=True,
            use_llm=True,   # vision calls slow down CI; disable for test
        ),
        timeout=600_000,
    )

    # ---------------- basic assertions ----------------
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert payload["count"] == 1
    assert len(payload["pages"])   == 1
    assert len(payload["filings"]) == 1

    # legacy page-stub still present
    first_page = payload["pages"][0]
    assert {"page", "pageImg", "totalPages", "rows"} <= first_page.keys()
    assert first_page["rows"] and "bbox" in first_page["rows"][0]

    # ---------------- schema validation ----------------
    validated = FilingModel.model_validate(payload["filings"][0]["filing"])
    assert validated.report_meta.candidate_name.value
