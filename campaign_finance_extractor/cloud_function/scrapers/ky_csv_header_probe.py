#!/usr/bin/env python3
"""
Quick sanity-check for KREF CSV exports.

• Contributions  → /ExportContributors
• Expenditures   → /Export

Query params are hard-wired to match the user-supplied curl examples:
    ElectionDate  = 11/5/2024   (or "11/5/2024 00:00:00" for expenditures)
    ElectionType  = GENERAL
    OfficeSought  = STATE REPRESENTATIVE
"""

from __future__ import annotations

import csv
import io
import logging
import sys
from textwrap import indent
from typing import List, Dict

import requests


ROOT = "https://secure.kentucky.gov/kref/publicsearch"
CONTRIB_EP = f"{ROOT}/ExportContributors"
EXPEND_EP  = f"{ROOT}/Export"

# ────────────────────────────────────────────────────────────────────────────
def fetch_csv(url: str, params: Dict[str, str]) -> List[Dict[str, str]]:
    """Download one CSV endpoint and return a list[dict] via csv.DictReader."""
    logging.info("GET %s", url)
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    # Export is UTF-8 with a BOM → strip it (decode("utf-8-sig"))
    content = resp.content.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(content)))


# ────────────────────────────────────────────────────────────────────────────
def show_sample(title: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print(f"\n‼️  {title}: 0 rows returned")
        return

    headers = list(rows[0].keys())
    print(f"\n🎯  {title} — CSV header row (canon):")
    print(indent("\n".join(headers), "  ├─ "))

    print(f"\n🔍  {title} — first 3 rows:")
    for r in rows[:3]:
        print(indent(str(r), "  • "))

    print(f"\n✅  {title}: total rows fetched = {len(rows):,}")


# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s ▶ %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # 1⃣  Contributions
    # ------------------------------------------------------------------
    contrib_params = {
        "FirstName": "",
        "LastName": "",
        "FromOrganizationName": "",
        "ElectionDate": "11/5/2024",
        "ElectionType": "GENERAL",
        "OfficeSought": "STATE REPRESENTATIVE",
        "Location": "",
        "City": "",
        "State": "",
        "Zip": "",
        "Employer": "",
        "Occupation": "",
        "OtherOccupation": "",
        "MinAmount": "",
        "MaxAmount": "",
        "MinimalDate": "",
        "MaximalDate": "",
        "ContributionMode": "",
        "ContributionSearchType": "All",
    }

    try:
        contrib_rows = fetch_csv(CONTRIB_EP, contrib_params)
        show_sample("Contributions", contrib_rows)
    except Exception as exc:
        logging.error("Contributions download/parse failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2⃣  Expenditures
    # ------------------------------------------------------------------
    expend_params = {
        "ElectionDate": "11/5/2024 00:00:00",   # portal sends full timestamp
        "ElectionType": "GENERAL",
        "OfficeSought": "STATE REPRESENTATIVE",
    }

    try:
        expend_rows = fetch_csv(EXPEND_EP, expend_params)
        show_sample("Expenditures", expend_rows)
    except Exception as exc:
        logging.error("Expenditures download/parse failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
