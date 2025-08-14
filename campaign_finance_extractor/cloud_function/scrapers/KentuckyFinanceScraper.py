# cloud_function/scrapers/KentuckyFinanceScraper.py
"""
Kentucky campaign‑finance scraper – writes both to Firestore **and** a local CSV
persister.  CSV rows follow the schema:

    Candidate_First_Last,Office,Election_Year,General_Or_Primary,
    Donation_Date,Donor_Name,Business_Donor_Address,Donor_City,Donor_State,
    Donation_Type,Amount,Donor_Type,Busines_Owner_Officer

Endpoints
---------
• /ExportContributors   → contributions CSV
• /Export               → expenditures  CSV
"""
from __future__ import annotations

import csv, io, logging, re, pathlib
from datetime import datetime
from typing import Dict, Any, List

import requests
from pydantic import BaseModel, Field, ValidationError
from google.cloud import firestore
from google.cloud.firestore_v1 import Increment  # type: ignore

from cloud_function.utils.firestore_utils import get_firestore_client

# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic row models
# ─────────────────────────────────────────────────────────────────────────────
class _ContributionRow(BaseModel):
    # —— required fields (parsed / normalised) ——
    amount: float
    receipt_date: str = Field(..., alias="date")
    candidate: str    = Field(..., alias="made_to")
    contributor: str
    # —— optional extras (kept if present) ——
    office_sought: str | None = None
    district: str      | None = None
    election_type: str | None = None
    contribution_type: str | None = None
    contribution_mode: str | None = None
    occupation: str | None = None
    employer: str   | None = None
    address_1: str  | None = None
    address_2: str  | None = None
    city: str       | None = None
    state: str      | None = None
    zip: str        | None = None
    in_kind_description: str | None = None
    statement_type: str | None = None


class _ExpenditureRow(BaseModel):
    amount: float
    expenditure_date: str = Field(..., alias="date")
    payee: str
    purpose: str | None = None
    office_sought: str | None = None
    district: str | None = None
    election_type: str | None = None
    address_1: str | None = None
    address_2: str | None = None
    city: str     | None = None
    state: str    | None = None
    zip: str      | None = None
    statement_type: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
class KentuckyFinanceScraper:
    """
    • `run_contributions()`  → writes raw rows **and** rolls up candidate + donor
      summaries, plus appends to a local CSV.
    • `run_expenditures()`   → writes raw rows (no summaries, no CSV).
    """

    ROOT        = "https://secure.kentucky.gov/kref/publicsearch"
    _CONTRIB_EP = f"{ROOT}/ExportContributors"
    _EXPEND_EP  = f"{ROOT}/Export"

    # CSV header – kept here for single‑point‑of‑truth
    CSV_HEADER = [
        "Candidate_First_Last",
        "Office",
        "Election_Year",
        "General_Or_Primary",
        "Donation_Date",
        "Donor_Name",
        "Business_Donor_Address",
        "Donor_City",
        "Donor_State",
        "Donation_Type",
        "Amount",
        "Donor_Type",
        "Busines_Owner_Officer",
    ]

    # ──────────────── init ───────────────────────────────────────────────
    def __init__(
        self,
        *,
        log_level: int = logging.INFO,
        csv_dir: str | pathlib.Path | None = None,
    ) -> None:
        """If *csv_dir* is provided, contribution rows will be appended to
        `csv_dir` / ky_contributions_<ELEC_DATE>_<TYPE>_<OFFICE>.csv
        """
        self.sess = requests.Session()
        self.sess.headers.update(
            {
                "User-Agent":
                "Mozilla/5.0 (scraper; +https://github.com/yourorg/ky-finance-bot)"
            }
        )

        self.log = logging.getLogger("ky")
        self.log.setLevel(log_level)
        if not self.log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s ▶ %(message)s", "%H:%M:%S")
            )
            self.log.addHandler(h)

        # —— one shared Firestore client across all batches ——
        self.db            = get_firestore_client()
        self.col_contrib   = self.db.collection("finance_contributions")
        self.col_expend    = self.db.collection("finance_expenditures")
        self.col_summary   = self.db.collection("finance_summary")
        self.col_donors    = self.db.collection("finance_donors")   # ← NEW

        # —— local CSV persister ——
        self.csv_dir: pathlib.Path | None = pathlib.Path(csv_dir).expanduser().resolve() if csv_dir else None
        if self.csv_dir:
            self.csv_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────── public API ─────────────────────────────────────────
    def run_contributions(self, *, election_date: str,
                          election_type: str, office: str) -> None:
        """
        Stream the KY “ExportContributors” CSV, write raw rows & update:

        • per‑candidate cycle roll‑ups  (finance_summary  collection)
        • per‑donor   aggregate totals  (finance_donors   collection)
        • append to local CSV           (if self.csv_dir is set)
        """
        params = {
            "FirstName": "", "LastName": "", "FromOrganizationName": "",
            "ElectionDate": election_date,
            "ElectionType": election_type,
            "OfficeSought": office,
            "Location": "", "City": "", "State": "", "Zip": "",
            "Employer": "", "Occupation": "", "OtherOccupation": "",
            "MinAmount": "", "MaxAmount": "", "MinimalDate": "",
            "MaximalDate": "", "ContributionMode": "",
            "ContributionSearchType": "All",
        }
        self._stream_csv(
            self._CONTRIB_EP,
            params,
            _ContributionRow,
            filing_type="contribution",
        )

    def run_expenditures(self, *, election_date: str,
                         election_type: str, office: str) -> None:
        params = {
            "FirstName": "", "LastName": "", "OrganizationName": "",
            "FromCandidateFirstName": "", "FromCandidateLastName": "",
            "FromOrganizationName": "",
            "ElectionDate": election_date,
            "ElectionType": election_type,
            "OfficeSought": office,
            "MinAmount": "", "MaxAmount": "", "MinimalDate": "",
            "MaximalDate": "", "PageSize": "", "PageIndex": "", "ReportId": "",
        }
        self._stream_csv(
            self._EXPEND_EP,
            params,
            _ExpenditureRow,
            filing_type="expenditure",
        )

    # ──────────────── core worker ───────────────────────────────────────
    def _stream_csv(
        self,
        endpoint: str,
        params: Dict[str, str],
        row_model: type[BaseModel],
        *,
        filing_type: str,
        batch_size: int = 500,
    ) -> None:
        """Common downloader / parser / Firestore‑writer (+ optional CSV)."""
        self.log.info("GET %s", endpoint)
        resp = self.sess.get(endpoint, params=params, timeout=120)
        resp.raise_for_status()

        rdr = csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig")))

        batch: List[Dict[str, Any]] = []
        csv_rows: List[List[str]] = []  # only for contributions
        total = 0

        for raw in rdr:
            try:
                base = self._canonicalise_row(raw)
                model_row = row_model(**base).model_dump(by_alias=True)
            except ValidationError as ve:
                self.log.debug("row skipped: %s", ve)
                continue

            enriched = self._augment(model_row, params, filing_type)
            batch.append(enriched)

            if filing_type == "contribution":
                csv_rows.append(self._as_csv_row(enriched))

            if len(batch) >= batch_size:
                self._commit(batch)
                total += len(batch)
                batch.clear()

        if batch:
            self._commit(batch)
            total += len(batch)

        # —— local CSV write (only for contributions) ——
        if filing_type == "contribution" and csv_rows and self.csv_dir:
            self._flush_csv(csv_rows, params)

        self.log.info("✅  %s %s rows written", total, filing_type)

    # ──────────────── helpers ───────────────────────────────────────────
    # ------------------------------------------------------------------
    #  CSV helpers
    # ------------------------------------------------------------------
    def _flush_csv(self, rows: List[List[str]], qs: Dict[str, str]) -> None:
        """Append *rows* to a CSV file under self.csv_dir."""
        assert self.csv_dir  # guaranteed by caller
        fname = (
            f"ky_contributions_"
            f"{qs['ElectionDate'].replace('/', '-')}_"
            f"{qs['ElectionType'].lower()}_"
            f"{self._slugify(qs['OfficeSought'])}.csv"
        )
        fpath = self.csv_dir / fname

        new_file = not fpath.exists()
        with fpath.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            if new_file:
                writer.writerow(self.CSV_HEADER)
            writer.writerows(rows)
        self.log.info("💾  appended %s rows to %s", len(rows), fpath)

    def _as_csv_row(self, r: Dict[str, Any]) -> List[str]:
        """Map *enriched contribution row* → ordered CSV fields."""
        address = " ".join(p for p in (r.get("address_1"), r.get("address_2")) if p)
        year = datetime.strptime(r["election_date"], "%m/%d/%Y").year
        return [
            r.get("candidate_display_name", ""),                # Candidate_First_Last
            r.get("office", ""),                                # Office
            str(year),                                           # Election_Year
            r.get("election_type", ""),                         # General_Or_Primary
            r.get("receipt_date", ""),                          # Donation_Date
            r.get("contributor", ""),                           # Donor_Name
            address,                                             # Business_Donor_Address
            r.get("city", ""),                                 # Donor_City
            r.get("state", ""),                                # Donor_State
            r.get("contribution_type") or r.get("contribution_mode", ""),  # Donation_Type
            f"{r['amount']:.2f}",                                # Amount
            r.get("contributor_type", ""),                     # Donor_Type
            r.get("occupation", ""),                           # Busines_Owner_Officer
        ]

    # ------------------------------------------------------------------
    #  Normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _slugify(txt: str) -> str:
        txt = re.sub(r"[^A-Za-z0-9]+", "-", txt.strip().lower())
        return re.sub(r"-{2,}", "-", txt).strip("-")

    def _augment(
        self,
        row: Dict[str, Any],
        qs: Dict[str, str],
        filing_type: str,
    ) -> Dict[str, Any]:
        """Add meta + roll‑up helper fields."""
        row.update(
            {
                "jurisdiction": "KY",
                "filing_type": filing_type,
                "election_date": qs["ElectionDate"],
                "election_type": qs["ElectionType"],
                "office": qs["OfficeSought"],
                "scraped_at": firestore.SERVER_TIMESTAMP,
                "source_url": self._CONTRIB_EP if filing_type == "contribution" else self._EXPEND_EP,
            }
        )

        # —— contribution‑specific enrichments ——
        if filing_type == "contribution":
            cand_name = row.get("made_to", "").strip()
            row["candidate_slug"]         = self._slugify(cand_name)
            row["candidate_display_name"] = cand_name

            year = datetime.strptime(qs["ElectionDate"], "%m/%d/%Y").year
            row["cycle_key"] = f"{year}_{qs['ElectionType'].upper()}"

            # naïve heuristic: anything that isn't explicitly "individual" is PAC/org
            ctype = (row.get("contribution_type") or "").lower()
            row["contributor_type"] = "individual" if "individual" in ctype else "pac/org"

            # for donor prefix search
            if row.get("contributor"):
                row["contributor_lc"] = row["contributor"].lower()

        return row

    # ------------------------------------------------------------------
    #  Canonicalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonicalise_row(raw: Dict[str, str]) -> Dict[str, Any]:
        """Fix typos, unify money/date fields & normalise keys."""
        r: Dict[str, Any] = dict(raw)  # shallow copy

        # —— money ──────────────────────────────────────────────────────
        money_key = next(k for k in ("Amount", "Dispursement Amount") if k in r)
        money_raw = r.pop(money_key, "").replace("$", "").replace(",", "")
        amt = float(money_raw.strip("()") or 0)
        if "(" in money_raw:  # negatives appear as ($1.23)
            amt *= -1
        r["amount"] = amt

        # —— date ───────────────────────────────────────────────────────
        for k in (
            "Receipt Date",
            "Contribution Date",
            "Expenditure Date",
            "Dispursement Date",
        ):
            if k in r:
                r["date"] = r.pop(k)
                break

        # —— simple equivalences / clean‑ups ────────────────────────────
        if "Location" in r:
            r["district"] = r.pop("Location")

        # recipient → made_to
        if "Recipient First Name" in r or "Recipient Last Name" in r:
            r["made_to"] = " ".join(
                [r.pop("Recipient First Name", ""), r.pop("Recipient Last Name", "")]
            ).strip()

        # contributor field (org OR FN/LN)
        if r.get("From Organization Name"):
            r["contributor"] = r.pop("From Organization Name")
        elif "Contributor First Name" in r or "Contributor Last Name" in r:
            r["contributor"] = " ".join(
                [r.pop("Contributor First Name", ""), r.pop("Contributor Last Name", "")]
            ).strip()

        # expenditure payee
        if "Organization Name" in r or "Recipient First Name" in r:
            payee = r.pop("Organization Name", "")
            payee = payee or " ".join(
                [r.pop("Recipient First Name", ""), r.pop("Recipient Last Name", "")]
            ).strip()
            if payee:
                r["payee"] = payee

        # lower‑case keys, spaces→underscores
        normalised: Dict[str, Any] = {}
        for k, v in r.items():
            normalised[k.lower().replace(" ", "_")] = v.strip() if isinstance(v, str) else v
        return normalised

    # ------------------------------------------------------------------
    #  Firestore write helpers
    # ------------------------------------------------------------------
    def _commit(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        is_contrib   = rows[0]["filing_type"] == "contribution"
        raw_col      = self.col_contrib if is_contrib else self.col_expend
        raw_batch    = self.db.batch()
        summary_batch: firestore.WriteBatch | None = self.db.batch() if is_contrib else None
        donor_batch  : firestore.WriteBatch | None = self.db.batch() if is_contrib else None

        # keep track so we only .set() each candidate once
        touched_candidates: set[str] = set()

        for r in rows:
            # 1️⃣  raw row
            raw_batch.set(raw_col.document(), r)

            if not is_contrib:
                continue           # expenditures don't roll‑up anything

            slug = r["candidate_slug"]
            cyc  = r["cycle_key"]
            cand_doc = self.col_summary.document(slug)

            # 2️⃣  candidate shell - set *once* per batch
            if slug not in touched_candidates:
                summary_batch.set(
                    cand_doc,
                    {
                        # static identity fields
                        "candidate_name": r["candidate_display_name"],
                        "office"        : r["office"],
                        "district"      : r.get("district"),
                        "jurisdiction"  : "KY",
                    },
                    merge=True,
                )
                touched_candidates.add(slug)

            # 3️⃣  roll‑up metrics
            incr: dict[str, Any] = {
                f"cycles.{cyc}.last_updated" : firestore.SERVER_TIMESTAMP,
                f"cycles.{cyc}.raised_total" : Increment(r["amount"]),
            }
            if r["contributor_type"] == "individual":
                incr[f"cycles.{cyc}.individual_total"] = Increment(r["amount"])
                incr[f"cycles.{cyc}.individual_count"] = Increment(1)
            else:
                incr[f"cycles.{cyc}.pac_total"] = Increment(r["amount"])

            summary_batch.update(cand_doc, incr)

            # 4️⃣  donor roll‑up (unchanged)
            donor_name = r.get("contributor") or ""
            if donor_name:
                donor_slug = self._slugify(donor_name)
                donor_doc  = self.col_donors.document(donor_slug)
                donor_batch.set(
                    donor_doc,
                    {
                        "name"         : donor_name,
                        "is_org"       : r["contributor_type"] != "individual",
                        "jurisdictions": firestore.ArrayUnion(["KY"]),
                        "total"        : Increment(r["amount"]),
                        "txns"         : Increment(1),
                        "updated_at"   : firestore.SERVER_TIMESTAMP,
                    },
                    merge=True,
                )

        # 5️⃣  commit everything
        raw_batch.commit()
        if summary_batch:
            summary_batch.commit()
        if donor_batch:
            donor_batch.commit()

        self.log.info("📝 wrote %s raw rows%s",
                      len(rows),
                      " + summaries + donor roll‑ups" if is_contrib else "")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI helper for.manual runs (local CSV path via --csv out)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="KY finance CSV scraper")
    ap.add_argument("--elect-date", default="11/5/2024")
    ap.add_argument("--elect-type", default="GENERAL")
    ap.add_argument("--office", default="STATE REPRESENTATIVE")
    ap.add_argument("--mode", choices=["contrib", "expend"], default="contrib")
    ap.add_argument("--csv-out", help="Directory to append local CSVs", default="./ky_csv")
    args = ap.parse_args()

    s = KentuckyFinanceScraper(csv_dir=args.csv_out)
    if args.mode == "contrib":
        s.run_contributions(
            election_date=args.elect_date,
            election_type=args.elect_type,
            office=args.office,
        )
    else:
        s.run_expenditures(
            election_date=args.elect_date,
            election_type=args.elect_type,
            office=args.office,
        )
