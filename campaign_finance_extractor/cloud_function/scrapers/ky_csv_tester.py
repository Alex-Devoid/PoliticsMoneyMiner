#!/usr/bin/env python3
"""
KY KREF CSV one-shot tester — fixed query string.

• Sends the exact same parameters as:
    https://secure.kentucky.gov/kref/publicsearch/AllContributors
  …but to /ExportContributors so we get CSV back.

• Prints a validation summary + first 5 rows.

Run:  python ky_csv_fixed.py
"""

from __future__ import annotations
import csv, io, logging, sys
from typing import Dict, List, Any

import requests
from pydantic import BaseModel, Field, ValidationError

# ───────────────────────────────────────────────────────────────────────────
ROOT   = "https://secure.kentucky.gov/kref/publicsearch"
ENDPT  = "ExportContributors"          # CSV endpoint

# Unmodified parameter list from the curl sample
PARAMS = {
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
    "PageSize": "10",         # kept even though CSV returns *all* rows
    "PageIndex": "0",
    "ReportId": "",
}

# ───────────────────────────────────────────────────────────────────────────
# Minimal Pydantic schema (tweak as you iterate)
class ContribCSVRow(BaseModel):
    amount: float
    date: str
    made_to: str | None = None
    contributor: str = Field(..., alias="contributor_name")
    occupation: str | None = None
    contrib_type: str | None = Field(None, alias="contribution_type")
    mode: str | None = Field(None, alias="contribution_mode")
    city: str | None = None
    state: str | None = None


# Helper to turn "$1,234.00" → 1234.0   and "($350.00)" → -350.0
def _dollars(raw: str) -> float:
    raw = raw.strip().replace("$", "").replace(",", "")
    sign = -1 if "(" in raw and ")" in raw else 1
    return sign * float(raw.strip("()") or 0)


def _clean(row: Dict[str, str]) -> Dict[str, Any]:
    """Map CSV headers into ContribCSVRow kwargs."""
    return {
        "amount": _dollars(row["Amount"]),
        "date": row["Date"],
        "made_to": row.get("Made To"),
        "contributor_name": row.get("Contributor Name", ""),
        "occupation": row.get("Occupation"),
        "contribution_type": row.get("Contribution Type"),
        "contribution_mode": row.get("Contribution Mode"),
        "city": row.get("City"),
        "state": row.get("State"),
    }


# ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s ▶ %(message)s",
        datefmt="%H:%M:%S",
    )

    url  = f"{ROOT}/{ENDPT}"
    logging.info("GET %s", url)
    resp = requests.get(url, params=PARAMS, timeout=60)
    resp.raise_for_status()

    buf = io.StringIO(resp.text, newline="")
    rdr = csv.DictReader(buf)

    good, bad = 0, 0
    preview: List[ContribCSVRow] = []
    for row in rdr:
        try:
            obj = ContribCSVRow(**_clean(row))
            good += 1
            if len(preview) < 5:
                preview.append(obj)
        except ValidationError as ve:
            bad += 1
            logging.debug("❌ row skipped: %s", ve.errors())

    print(f"\n✔️  validated {good:,} rows (skipped {bad:,}) — preview:")
    for r in preview:
        print("   ", r.model_dump_json())


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        logging.error("HTTP %s\n%s", e.response.status_code, e.response.text[:400])
        sys.exit(1)
