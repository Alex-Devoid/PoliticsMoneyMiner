# cloud_function/routes/sedgwick.py
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import hashlib
import time
import random
import datetime as _dt
from typing import Optional, List, Dict, Any

from cloud_function.scrapers.SedgwickElectionsCampaignExpenseReport import (
    SedgwickExpenseScraper,
)

router = APIRouter(prefix="/sedgwick", tags=["sedgwick"])

# ───────────────────────────────────────── helpers ──────────────────────────
def _filing_to_page_stub(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one filing-level record into a single-page stub that satisfies the
    legacy test harness.  Ensures **every** row carries a ``bbox`` key.
    """
    filing = rec.get("filing") or {}

    # 1️⃣ choose rows – parsed rows ➜ raw tokens ➜ []
    rows: List[dict] = filing.get("rows") or filing.get("tokens", [])

    # 2️⃣ guarantee bbox
    patched: List[dict] = []
    for r in rows:
        if "bbox" not in r:
            bbox = [r.get("x0"), r.get("y0"), r.get("x1"), r.get("y1")]
            r = dict(r) | {"bbox": bbox}
        patched.append(r)

    return {
        "page": 1,
        "pageImg": "",
        "totalPages": 1,
        "rows": patched,
    }


# ───────────────────────────────────────── endpoint ─────────────────────────
@router.post("/scrape")
async def scrape_expense_reports(
    # —— original crawl params ————————————————————————————————
    year: int = 0,
    office: Optional[str] = None,
    # —— local-file override ————————————————————————————————
    local_pdf: Optional[str] = None,
    # —— processing flags ————————————————————————————————
    ocr: bool = True,
    use_llm: bool = True,
    save_pdfs: bool = True,      # 👈 default ON so artefacts are written
    # —— crawl niceties ————————————————————————————————
    limit: Optional[int] = 20,
    cool_down: float = 0.4,
):
    """
    • When *local_pdf* is supplied we **skip** the public portal and process just
      that file; other query-string params (year, office, limit…) are ignored.

    • If *save_pdfs* is true we always persist a matching *.json* payload and,
      when bytes are available, the *.pdf* into **tests/scrapers/**.
    """
    t0 = _dt.datetime.utcnow()

    scraper = SedgwickExpenseScraper(
        ocr=ocr,
        use_openai=use_llm,
        timeout=60_000,
    )

    # ─────────────── single-file mode ───────────────
    if local_pdf:
        pdf_path = Path(local_pdf).expanduser().resolve()
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

        pdf_bytes   = pdf_path.read_bytes()
        filing_dict = scraper._build_filing_payload(pdf_bytes)

        filings: List[Dict[str, Any]] = [{
            "doc_id"   : pdf_path.name,
            "office"   : office or "LOCAL",
            "metadata" : {},
            "pdf_bytes": pdf_bytes,
            "filing"   : filing_dict,
        }]

    # ─────────────── crawl mode ───────────────
    else:
        filings = []
        for idx, rec in enumerate(scraper.run(year, office), start=1):
            filings.append(rec)
            if limit and idx >= limit:
                break
            time.sleep(cool_down + random.uniform(0, cool_down * 0.2))

        if not filings:
            raise HTTPException(status_code=404, detail="No filings found")

    elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
    print(f"[DBG] collected {len(filings)} filing(s) in {elapsed:.1f}s")

    # ─────────────────────────── persist artefacts (pdf/json) ─────────────────
    # ------------------------------------------------------------------ PDF dump
    saved_dir: Optional[str] = None
    if save_pdfs:
        out_dir = Path("tests/scrapers").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for r in filings:
            # ---------------------------------------------------------- helpers
            fname_root = hashlib.sha1(str(r["doc_id"]).encode()).hexdigest()

            # ---------- 1 · raw PDF ------------------------------------------
            if pdf := r.pop("pdf_bytes", None):                # strip bytes
                (out_dir / f"{fname_root}.pdf").write_bytes(pdf)

            # ---------- 2 · structured Filing (schema-conformant) ------------
            filing_struct = {k: v for k, v in r["filing"].items()
                             if not k.startswith("_")}          # drop helper keys
            (out_dir / f"{fname_root}.json").write_text(
                json.dumps(filing_struct, indent=2, ensure_ascii=False)
            )

            # ---------- 3 · initial free-form JSON from o3 --------------------
            init_free = r["filing"].get("_initial_llm_json")
            if init_free is not None:
                (out_dir / f"{fname_root}.initial.json").write_text(
                    json.dumps(init_free, indent=2, ensure_ascii=False)
                )

        saved_dir = str(out_dir)
        print(f"[DBG]   wrote artefacts → {saved_dir}")

    # ─────────────────────────── response munging ────────────────────────────
    pages_view   = [_filing_to_page_stub(r) for r in filings]
    filings_view = [{k: v for k, v in r.items() if k != "pdf_bytes"} for r in filings]

    return {
        "year"       : year,
        "office"     : office,
        "count"      : len(pages_view),
        "elapsed_sec": round(elapsed, 2),
        "pages"      : pages_view,
        "filings"    : filings_view,
    }
