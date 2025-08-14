# tests/scrapers/test_sedgwick_10_live.py
"""
Full-stack smoke-test for the Sedgwick scraper.

• Points Firestore *and* Cloud-Storage client-libs at the emulators
• Streams real PDFs from Sedgwick
• Saves PDF + parsed JSON artefacts to ./sedgwick_raw
• Uses a tiny SafeScraper subclass so the test runs even when the
  first-pass LLM is disabled (use_openai=False).
"""
from __future__ import annotations

import os, sys, json, pathlib, logging
from pathlib import Path
from typing import Set

# ────────────────────────────────
# 1️⃣  Emulator environment
# ────────────────────────────────
os.environ["FIRESTORE_EMULATOR_HOST"] = "firestore-emulator:8080"
os.environ["GOOGLE_CLOUD_PROJECT"]    = "dev-emulated-project"
os.environ["FIRESTORE_DATABASE_ID"]   = "(default)"
os.environ["STORAGE_EMULATOR_HOST"]   = "http://firebase-emulator:9199"   # Firebase Storage

print(
    "Environment Variables\n"
    f"  FIRESTORE_EMULATOR_HOST = {os.getenv('FIRESTORE_EMULATOR_HOST')}\n"
    f"  STORAGE_EMULATOR_HOST   = {os.getenv('STORAGE_EMULATOR_HOST')}\n"
    f"  GOOGLE_CLOUD_PROJECT    = {os.getenv('GOOGLE_CLOUD_PROJECT')}\n"
)

# ────────────────────────────────
# 2️⃣  Local import – after env-vars
# ────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from cloud_function.scrapers.SedgwickElectionsCampaignExpenseReport import (  # noqa: E402
    SedgwickExpenseScraper,
)

# ────────────────────────────────
# 3️⃣  Safe wrapper so CI runs
# ────────────────────────────────
class SafeScraper(SedgwickExpenseScraper):
    """Patch _attach_bboxes_heuristic so it tolerates use_openai=False."""
    def _attach_bboxes_heuristic(self, initial_json, rows_by_pg, fuzz=0.8):
        if not initial_json:                # first-pass LLM skipped
            return {}                       # simply return empty dict
        return super()._attach_bboxes_heuristic(initial_json, rows_by_pg, fuzz)

# ────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

YEAR           = 2024
OFFICE_FILTER  = "COUNTY COMMISSIONER"
MAX_CANDIDATES = 5

scraper = SafeScraper(
    ocr=True,
    use_openai=True,   # keep CI completely offline / fast
    timeout=20,
)

out_dir = pathlib.Path("sedgwick_raw")
out_dir.mkdir(exist_ok=True)

seen: Set[str] = set()

for rec in scraper.run(YEAR, OFFICE_FILTER):
    slug = (
        rec["filing"]["report_meta"]["candidate_name"]["value"]
        .lower()
        .replace(" ", "-")
    )
    if slug in seen:
        continue
    seen.add(slug)

    # save artefacts for manual inspection / diffs
    (out_dir / f"{slug}.pdf").write_bytes(rec["pdf_bytes"])
    (out_dir / f"{slug}.json").write_text(json.dumps(rec["filing"], indent=2))

    print(f"✔  {slug:<25} ({len(seen):2}/{MAX_CANDIDATES})")

    if len(seen) >= MAX_CANDIDATES:
        break

assert len(seen) == MAX_CANDIDATES, (
    f"Expected {MAX_CANDIDATES} candidates, got {len(seen)}"
)
