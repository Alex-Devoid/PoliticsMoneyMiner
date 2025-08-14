"""
cloud_function/routes/kentucky.py
---------------------------------
FastAPI wrapper around KentuckyFinanceScraper.

• POST /kentucky/finance – scrape CSV export and write to Firestore (and, if
  KYFIN_CSV_DIR is set, append to a local CSV)
"""
from __future__ import annotations

import os
import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, status
from pydantic import BaseModel, constr

from cloud_function.scrapers.KentuckyFinanceScraper import KentuckyFinanceScraper

router = APIRouter(prefix="/kentucky", tags=["kentucky"])
_log = logging.getLogger("routes.ky")

# ── request / response models ─────────────────────────────────────────
class ScrapeRequest(BaseModel):
    election_date: constr(strip_whitespace=True) = "11/5/2024"   # mm/dd/yyyy
    election_type: constr(strip_whitespace=True) = "GENERAL"     # e.g. PRIMARY
    office       : constr(strip_whitespace=True) = "STATE REPRESENTATIVE"
    mode         : Literal["contrib", "expend"] = "contrib"

class ScrapeResponse(BaseModel):
    detail: str
    rows_enqueued: int

# ── endpoint ──────────────────────────────────────────────────────────
@router.post(
    "/finance",
    response_model=ScrapeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def scrape_finance(req: ScrapeRequest, bg: BackgroundTasks):
    """Kick off a Kentucky CSV scrape in the background."""

    # honour KYFIN_CSV_DIR if set (tests & local runs rely on this)
    csv_dir = os.getenv("KYFIN_CSV_DIR")
    scraper = KentuckyFinanceScraper(csv_dir=csv_dir)

    def _run() -> None:
        _log.info("KY scrape started: %s", req.json())
        if req.mode == "contrib":
            scraper.run_contributions(
                election_date=req.election_date,
                election_type=req.election_type,
                office=req.office,
            )
        else:
            scraper.run_expenditures(
                election_date=req.election_date,
                election_type=req.election_type,
                office=req.office,
            )
        _log.info("KY scrape finished.")

    bg.add_task(_run)
    return ScrapeResponse(detail="Scrape scheduled", rows_enqueued=0)
