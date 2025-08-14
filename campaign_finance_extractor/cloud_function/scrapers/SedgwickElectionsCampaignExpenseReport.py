"""
Sedgwick County – Campaign-Finance scraper + processor
⇢ Minimal test harness for the 2023 *Wichita City Mayor* race, or any single
  office you pass on the CLI.  The scraper still supports crawling **all**
  offices when you omit --office.
"""
from __future__ import annotations
import re, difflib
from collections import defaultdict
# ── stdlib ────────────────────────────────────────────────────────────────────
import io, base64, os, json, logging, pathlib, argparse, hashlib
from json import JSONDecodeError
import uuid, datetime, tempfile  
from typing import Iterator, List, Dict, Any, Optional
from typing_extensions import TypedDict          # ← Pydantic‑safe TypedDict


# ── 3rd‑party ────────────────────────────────────────────────────────────────
from PIL import Image,ImageOps, ImageFilter
import re
import fitz                                       # PyMuPDF
import pdf2image, pytesseract                     # populated by Docker layer
import requests
from requests import Session, exceptions as rex
from rapidfuzz import fuzz
from google.cloud import firestore, storage
from google.cloud.firestore_v1 import Increment
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from pydantic import BaseModel, Field, Extra
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from cloud_function.scrapers.handwriting_ocr import ocr_handwritten
from cloud_function.scrapers.ocr_backends import trocr_text
from cloud_function.utils.firestore_utils import get_firestore_client
from cloud_function.utils.storage_utils import get_storage_client


from dotenv import load_dotenv 
load_dotenv()
__all__ = ["SedgwickExpenseScraper"]
TESS_CFG = (
    "--oem 1 --psm 6 "
    "-c tessedit_char_whitelist="
    "0123456789$.,:/-%ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
# ─────────────────────────────────────────────────────────────────────────────
class SedgwickExpenseScraper:
    """Scraper façade – create, then `.run(year, office?)` to get one Filing dict."""

    ROOT = "https://imaging.sedgwickcounty.org"
    UA   = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    )

    # ── init ───────────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        ocr: bool                = True,
        use_openai: bool         = False,
        openai_model: str        = "o3",
        openai_api_key: str | None = None,
        timeout: int             = 15,
        log_level: int           = logging.INFO,
    ) -> None:

        # basic HTTP session --------------------------------------------------
        self.sess = Session()
        self.sess.headers.update({
            "User-Agent"      : self.UA,
            "Accept"          : "application/json, text/plain, */*",
            "Accept-Language" : "en-US,en;q=0.9",
            "Origin"          : self.ROOT,
            "Referer"         : f"{self.ROOT}/PublicAccess/index.html",
            "Connection"      : "keep-alive",
        })
        self.timeout  = timeout
        self.ocr      = ocr and pdf2image is not None and pytesseract is not None
        self.use_llm  = use_openai and ChatOpenAI is not None

        fmt = "%(asctime)s %(levelname)s %(name)s:%(lineno)d ▶ %(message)s"
        self.log = logging.getLogger("sedgwick")
        self.log.setLevel(log_level)

        if not any(isinstance(h, logging.StreamHandler) for h in self.log.handlers):
            h = logging.StreamHandler()        # writes to stdout (pytest -s shows it)
            h.setLevel(log_level)              # honour constructor arg
            h.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
            self.log.addHandler(h)

        self.log.propagate = False             # avoid double-logging via root

        self.db = get_firestore_client()

        self.storage_client = get_storage_client()
            # Production / real Google Cloud Storage
            
        self.col_contrib  = self.db.collection("finance_contributions")
        self.col_expend   = self.db.collection("finance_expenditures")
        self.col_summary  = self.db.collection("finance_summary")
        self.col_donors   = self.db.collection("finance_donors")
        self.col_files    = self.db.collection("finance_files")   # NEW
        self._SLUG_PAT = re.compile(r"[^a-z0-9]+")
        self.storage_client = storage.Client()
        self.bucket_name    = os.getenv("SEDGWICK_BUCKET", "sedgwick-finance-files")
        self.bucket         = self.storage_client.bucket(self.bucket_name)        

        # a dedicated child logger for the first-pass LLM if you want it
        self.o3_log = logging.getLogger("sedgwick.o3")
        self.o3_log.setLevel(logging.DEBUG)    # keep it chatty
        self.o3_log.propagate = False
        if not any(isinstance(h, logging.StreamHandler) for h in self.o3_log.handlers):
            self.o3_log.addHandler(h)      

        # optional LLM client -------------------------------------------------
        # ---------------------------------------------------------------- model clients
        self.o3_llm  = None          # first-pass “free-form” extractor
        self.llm     = None          # second-pass structured extractor
        if self.use_llm:
            self.o3_llm = ChatOpenAI(          # ← always o3
                model      = "gpt-4o",
                api_key    = openai_api_key or os.getenv("OPENAI_API_KEY"),
                temperature= 0.7,              # deterministic pass
            )
            self.o3_llm
            self.llm = ChatOpenAI(             # ← configurable (defaults to gpt-4o-mini)
                model      = openai_model,
                api_key    = openai_api_key or os.getenv("OPENAI_API_KEY")                
            )

    # ── pydantic schemas --------------------------------------------------
        # ── pydantic schemas --------------------------------------------------
        class FieldBBox(BaseModel):
            value: str | float | int | None
            bbox: list[float] | None

        class ReportMeta(BaseModel):
            candidate_name: FieldBBox
            address: FieldBBox
            city: FieldBBox
            state: FieldBBox
            zip_code: FieldBBox
            county: FieldBBox
            office_sought: FieldBBox
            district: FieldBBox
            report_date: FieldBBox
            period_start: FieldBBox
            period_end: FieldBBox
            cash_on_hand_beginning: FieldBBox
            total_contributions_receipts: FieldBBox
            cash_available: FieldBBox
            total_expenditures_disbursements: FieldBBox
            cash_on_hand_closing: FieldBBox
            signature_name: FieldBBox
            signature_date: FieldBBox

        class ScheduleARow(BaseModel):
            date: str
            contributor: str
            address: str
            occupation: str
            type: str
            amount: float
            bbox: list[float]

        class ScheduleCRow(BaseModel):
            date: str
            payee: str
            address: Optional[str]
            purpose: str
            amount: float
            bbox: list[float]

        class Filing(BaseModel):
            report_meta: ReportMeta
            schedule_a: list[ScheduleARow]
            schedule_c: list[ScheduleCRow]




        class _Cand(BaseModel):
            name: str = Field(..., description="candidate’s full name")
            address: str | None = None
            city: str | None = None
            zip: str | None = None
            county: str | None = None
            office_sought: str | None = None
            district: str | None = None

        class _Summary(BaseModel):
            cash_on_hand_beginning: float | None = None
            total_contributions: float | None = None
            cash_available: float | None = None
            total_expenditures: float | None = None
            cash_on_hand_close: float | None = None
            in_kind_contributions: float | None = None
            other_transactions: float | None = None

        class _Contrib(BaseModel):
            date: str
            contributor: str
            address: str | None = None
            occupation: str | None = None
            amount: float

        class _Expend(BaseModel):
            date: str
            name: str
            address: str | None = None
            purpose: str | None = None
            amount: float

        class _InKind(BaseModel):
            date: str
            contributor: str
            description: str | None = None      # “Purpose” column
            value: float                        # dollar value (number, not string)

        class _Debt(BaseModel):
            date: str
            creditor: str
            purpose: str | None = None
            balance: float            

        class _InitialExtract(BaseModel):
            candidate: _Cand
            summary: _Summary
            contributions: list[_Contrib] = []
            total_itemized_receipts: float | None = None
            total_unitemized_contributions: float | None = None
            expenditures: list[_Expend] = []
            total_itemized_expenditures: float | None = None
            total_unitemized_expenditures: float | None = None
            other_transactions: list[dict] = []
            in_kind_contributions: list[_InKind] = []   # ⇠ schedule B
            other_transactions   : list[_Debt]   = []

        _INITIAL_PARSER = PydanticOutputParser(pydantic_object=_InitialExtract)
        self._initial_parser = _INITIAL_PARSER
        self._FORMAT_HINT     = _INITIAL_PARSER.get_format_instructions()            

        self.Filing         = Filing            # expose for IDEs / type‑hints
        self.filing_parser  = PydanticOutputParser(pydantic_object=Filing)

        # ---- LLM prompts ----------------------------------------------------
        self.INITIAL_PROMPT_TEMPLATE = (
            """
            Using the best method available to you, extract *all* campaign‑finance
            information from the attached PDF filing.  Return **only** a JSON
            object – no Markdown, no comments.  The structure should reflect
            the data naturally present in the document (candidate details,
            report summary, schedules, etc.) but **do not conform to any rigid
            pre‑defined schema**.  Include an additional key called
            "extraction_workflow" that briefly explains, in 3‑6 bullet points,
            the steps you took (e.g. reading order analysis, table detection,
            line‑item grouping).

            Begin.
            """
        )

        class _ExpendEnriched(BaseModel):
            date: str
            name: str
            address: str | None = None
            purpose: str | None = None
            amount: float
            bbox: list[float] | None = None
            _page: int | None = None            # added so we can persist page-hints

        class _InitialEnriched(_InitialExtract):        # reuse your first-pass schema
            expenditures: list[_ExpendEnriched] = []     


        # new helper model — keeps the parts we care about and
        # lets everything else pass straight through (extra = "allow")
        class _BBoxExpend(BaseModel):
            date: str
            name: str
            address: str | None = None
            purpose: str | None = None
            amount: float
            bbox: list[float] | None = None
            _page: int | None = None

        class _EnrichedInitial(BaseModel):
            # we only validate the tricky list; every other key is accepted verbatim
            expenditures: list[_BBoxExpend]
            class Config:  # allow the many other keys from initial_json
                extra = Extra.allow


        self._enrich_parser = PydanticOutputParser(
            pydantic_object=_InitialEnriched
        )               


        _ENRICH_TEMPLATE = """
        # Goal:
        Your goal is to enrich the campaign_json objects with bboxes constructed from OCR tokens:        
        You are given two inputs:

        1. **campaign_json** – JSON already extracted from the filing  
        2. **tokens**       – OCR tokens for *one* slice of pages

        * Note: When matching campaign_json with OCR tokens, you will need to interpolate how poor OCR resulsts would match to the extracted campaign_json data. Keep in mind that both are coming from the same original source.

        campaign_json = {initial_json}
        tokens       = {tokens_json}

        Each token is an object like:

        ```

        {{
        "text": "<str>",
        "page": <int>,
        "x0": <float>, "y0": <float>,
        "x1": <float>, "y1": <float>
        }}

        ```

        ## TASK:
        For every *scalar* value in **initial_json** that appears **in this slice
        only**, find the best-matching token(s) and add a sibling key  

        ```

        "bbox": \[x0, y0, x1, y1]

        ```


        Return **only** the complete JSON – no markdown, comments, or extra text.
        Return strict JSON:
        {format_instructions}
        """.lstrip()
        


        # ── build inside __init__() ─────────────────────────────────────────────
        # _ENRICH_TEMPLATE = PromptTemplate("""
        # # Goal:
        # Your goal is to enrich the campaign_json objects with bboxes constructed from OCR tokens:
        
        # You are given **one slice** of an OCR’d filing.

        # • `initial_json` – previously-extracted data (may already contain some "bbox")
        # • `tokens`       – OCR rows for the current slice only

        # campaign_json = {initial_json}
        # tokens       = {tokens_json}

        # Each token row is:
        # { "text": str, "page": int, "x0": float, "y0": float, "x1": float, "y1": float }

        # # TASK
        # For every scalar value that appears in `initial_json` **and** is visible inside
        # `tokens`, add or keep a sibling key:

        #     "bbox": [x0, y0, x1, y1]

        # • Use the *row* bbox that covers the whole value (not individual characters).  
        # • Do **not** invent values that are absent from this page slice.
        # Return strict JSON:
        # {{format_instructions}}

        # Return the FULL JSON object *only* – no markdown, no comments.
        # """


        self.enrich_prompt = PromptTemplate(
            template=_ENRICH_TEMPLATE,
            input_variables=["initial_json", "tokens_json"],
            partial_variables={"format_instructions": self._enrich_parser.get_format_instructions()},
        )



   


    

        FILING_PROMPT = (
            """
            # INPUT

            tokens = {tokens_json}

            Each token is an object:
            { "text": str, "page": int, "x0": float, "y0": float, "x1": float, "y1": float }
            Coordinates are in native PDF pixels (origin top‑left).

            # OUTPUT

            Return **one** JSON object only — no Markdown, no comments — with exactly these
            top‑level keys:

            • "report_meta"   • "schedule_a"   • "schedule_c"

            ---------------------------------------------------------------------
            1. report_meta  (single object)
            ---------------------------------------------------------------------
            Every label below must itself be an object of the form
            "<label>": { "value": <string|number>, "bbox": [x0,y0,x1,y1] | null }

            candidate_name
            address
            city
            state
            zip_code
            county
            office_sought
            district
            report_date                 (YYYY-MM-DD)
            period_start                (YYYY-MM-DD)
            period_end                  (YYYY-MM-DD)
            cash_on_hand_beginning      (number)
            total_contributions_receipts (number)
            cash_available              (number)
            total_expenditures_disbursements (number)
            cash_on_hand_closing        (number)
            signature_name
            signature_date              (YYYY-MM-DD)

            ---------------------------------------------------------------------
            2. schedule_a  (array of contribution rows)
            ---------------------------------------------------------------------
            Each row object must contain:
            date         (YYYY-MM-DD)
            contributor  (full name or org)
            address      (street, city ST ZIP)
            occupation   (or industry)
            type         ("check", "e-funds", etc.)
            amount       (number)
            bbox         [x0,y0,x1,y1]   ← rectangle enclosing *all* tokens for that row

            ---------------------------------------------------------------------
            3. schedule_c  (array of expenditure rows)
            ---------------------------------------------------------------------
            Each row object must contain:
            date         (YYYY-MM-DD)
            payee        (name)
            address      (street, city ST ZIP, if present)
            purpose      (free-text)
            amount       (number)
            bbox         [x0,y0,x1,y1]

            ---------------------------------------------------------------------
            RULES
            ---------------------------------------------------------------------
            1. **Best‑match mapping** – choose the most accurate tokens for every field.
            2. **Consolidate split amounts** – e.g. "$ 1 234 . 56" → 1234.56
            3. **Numbers** – output JSON numbers only (no $, commas, or quotes).
            4. **bbox** – must enclose *all* tokens used; if field absent, set "bbox": null.
            5. If a required field is missing, use "" (or 0 for numbers) and bbox = null.
            6. Ignore extraneous text such as footers, page numbers, or fax headers.
            7. Output must be valid JSON and **nothing else**.

            Begin.
            """
        )

        self.filing_prompt = PromptTemplate(
            template=FILING_PROMPT,
            input_variables=["tokens_json"],
        )


    def _attach_bboxes_heuristic(self,initial_json: dict,
                                rows_by_pg: dict[int, list[dict]],
                                fuzz=0.8) -> dict:
        """
        For every expenditure in `initial_json["expenditures"]`
        try to attach a bbox by fuzzy-matching the row text.
        """
        if "expenditures" not in initial_json:
            return initial_json

        for exp in initial_json["expenditures"]:
            if "bbox" in exp and exp["bbox"]:
                continue                       # already done

            tgt_name  = re.sub(r"[^A-Za-z0-9]", " ", exp["name"]).lower()
            tgt_amt   = f"{exp['amount']:.2f}" if exp.get("amount") else None

            best   = None
            best_r = None
            for rows in rows_by_pg.values():
                for r in rows:
                    txt = r["text"].lower()
                    score = difflib.SequenceMatcher(None, tgt_name, txt).ratio()
                    if tgt_amt and tgt_amt not in txt:
                        score -= 0.1           # require amount to appear
                    if score > fuzz and (not best or score > best):
                        best, best_r = score, r

            if best_r:
                exp["bbox"] = best_r["bbox"]
                exp["_page"] = best_r["page"]

        return initial_json

    # ────────────────────────────────────────────────────────────────────
    #  First-pass vision LLM (per-page) – now with domain hints
    # ────────────────────────────────────────────────────────────────────
    def _initial_llm_extract(self, page_images: list[Image.Image]) -> dict | None:
        """
        Run GPT‑4o on **each** PDF page.  Every extracted sub‑object gets a `_page`
        hint so downstream heuristic matching has context.  If the model fails on
        a page we skip it – the rest of the pipeline can still succeed.
        """
        print("test test test")
        if not (self.use_llm and self.o3_llm):
            self.o3_log.debug("LLM disabled – skipping first‑pass extract")
            print("LLM disabled – skipping first‑pass extract")
            return None

        merged: dict = {}

        # ---------- common text template (re‑used for every page) --------------
        PAGE_PROMPT = f"""
        You are reading a **Kansas municipal campaign‑finance filing**.  
        Each page can contain one (or none) of the following sections:

        • **Header / Summary** – candidate details and a 6‑line numeric roll‑up  
        • **Schedule A** – “Contributions and Other Receipts”  
        • **Schedule B** – “In‑Kind Contributions”  
        • **Schedule C** – “Expenditures and Other Disbursements”  
        • **Schedule D** – “Other Transactions / Loans / Debts”

        Typical column headings you may see:
        Date · Name · Address · Occupation · Type · Amount · Purpose · Value ·
        Balance · Nature of Account

        **TASK**

        * Extract **only the information visible on this page**.  
        * Return a JSON object that conforms to this schema:

        {self._FORMAT_HINT}

        * Use JSON numbers for money (no `$`, commas, or quotes).  
        * If this page contains none of the above sections, return an empty
        object (`{{}}`).  
        * Output **pure JSON** – **no Markdown**, no commentary.
        """
        print("start initial extract")
        for pg_no, img in enumerate(page_images, start=1):
            try:
                # ---------- build multimodal message --------------------------
                buf = io.BytesIO(); img.save(buf, "PNG")
                mm_msg = [
                    {"type": "image_url",
                    "image_url": {"url": "data:image/png;base64," +
                                            base64.b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": PAGE_PROMPT},
                ]

                raw = self.o3_llm.invoke([HumanMessage(content=mm_msg)]).content
                print(raw)
                # Try a strict parse first; fall back to plain json.loads + re‑dump
                try:
                    page_obj = self._initial_parser.parse(raw)
                except Exception:
                    page_obj = self._initial_parser.parse(
                        json.dumps(json.loads(raw))   # tolerate spacing / newlines
                    )

                page_data = page_obj.model_dump()
                print("page data")
                print(page_data)
                # ---------- tag every top‑level dict with page number ----------
                for k, v in page_data.items():
                    if isinstance(v, dict):
                        v["_page"] = pg_no
                    elif isinstance(v, list):
                        for row in v:
                            if isinstance(row, dict):
                                row["_page"] = pg_no

                merged = self._merge_initial_pages(merged, page_data)

                # quick artefact for offline QC
                (pathlib.Path("tests/scrapers") /
                f"o3_p{pg_no}.json").write_text(
                    json.dumps(page_data, indent=2, ensure_ascii=False)
                )

            except Exception as exc:
                self.o3_log.warning("page %s extract failed (%s) – skipping", pg_no, exc)
                print(exc)
                continue

        # --------- don’t blow up the run if the model gave us nothing ----------
        if not merged:
            self.o3_log.warning("LLM produced no extractable JSON; continuing with heuristics")
            return None

        pathlib.Path("tests/scrapers/o3_initial.latest.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False)
        )
        return merged



    # ---------- HTTP helpers --------------------------------------------------
    def _get(self, url: str, **kw):
        try:
            resp = self.sess.get(url, timeout=self.timeout, **kw)
            resp.raise_for_status()
            return resp
        except rex.RequestException as exc:
            self.log.warning("GET %s error: %s", url, exc)
            return None

    def _post_json(self, url: str, payload: Dict[str, Any]):
        try:
            resp = self.sess.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except rex.RequestException as exc:
            self.log.warning("POST %s error: %s", url, exc)
            return None

    # ---------- pdf helpers --------------------------------------------------
    def _pdf_pages_to_images(self, pdf_bytes: bytes) -> list[Image.Image]:
        """300-dpi PIL images with contrast/threshold to boost handwriting."""
        pages = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
        proc  = []
        for p in pages:
            gray   = ImageOps.grayscale(p)
            gray   = ImageOps.autocontrast(gray, cutoff=1)      # clip 1 %
            blur   = gray.filter(ImageFilter.MedianFilter(size=3))
            thresh = blur.point(lambda v: 255 if v > 180 else 0)
            proc.append(thresh)
        return proc

    def _extract_typed_spans(self, pdf_bytes: bytes) -> List[dict]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        spans = []
        for pageno, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        spans.append({
                            "page": pageno,
                            "text": span["text"],
                            "x0": span["bbox"][0],
                            "y0": span["bbox"][1],
                            "x1": span["bbox"][2],
                            "y1": span["bbox"][3],
                        })
        return spans

    

    # ── token-to-row bucketing helper ──────────────────────────────────────
    def _group_rows(self,
                    tokens: list[dict],
                    y_tol: int = 6) -> list[dict]:
        """
        Collapse word-level tokens into visual “rows”.

        Parameters
        ----------
        tokens : list[dict]
            Each dict *must* contain the keys
            ``page, text, x0, y0, x1, y1`` (as produced by
            `_merge_tokens()`).

        y_tol : int, optional
            Allowed vertical jitter (px).  Two tokens whose *vertical
            centres* differ by ≤ ``y_tol`` are considered part of the
            same row.  Tweak if you notice over- or under-merging.

        Returns
        -------
        list[dict]
            One item per detected row::

                {
                    "page"  : int,                  # 1-based
                    "text"  : str,                  # L→R concatenation
                    "bbox"  : [x0, y0, x1, y1],     # row rectangle
                    "tokens": list[dict],           # source words
                }
        """

        if not tokens:
            return []

        # Stable sort → page ▸ vertical centre ▸ x-pos
        tokens = sorted(
            tokens,
            key=lambda t: (
                t["page"],
                (t["y0"] + t["y1"]) / 2,   # row baseline
                t["x0"],
            ),
        )

        rows: list[dict] = []
        current: list[dict] = []
        cur_page: Optional[int] = None
        cur_y: float | None = None

        def _flush():
            """Emit the current row into *rows*."""
            if not current:
                return
            xs = [w["x0"] for w in current] + [w["x1"] for w in current]
            ys = [w["y0"] for w in current] + [w["y1"] for w in current]
            rows.append(
                {
                    "page"  : current[0]["page"],
                    "text"  : " ".join(
                        w["text"] for w in sorted(current, key=lambda w: w["x0"])
                    ),
                    "bbox"  : [min(xs), min(ys), max(xs), max(ys)],
                    "tokens": current[:],
                }
            )
            current.clear()

        # --- main pass --------------------------------------------------
        for tok in tokens:
            centre_y = (tok["y0"] + tok["y1"]) / 2

            if (
                cur_page is None                       # first token
                or tok["page"] != cur_page             # new page
                or abs(centre_y - (cur_y or 0)) > y_tol  # new row
            ):
                _flush()
                cur_page = tok["page"]
                cur_y = centre_y

            current.append(tok)

        _flush()  # last row
        return rows

    
    # ------------------------------------------------------------------
    #  OCR (Tesseract ➜ optional TrOCR fallback)
    # ------------------------------------------------------------------
    def _extract_ocr_spans(self, images: list[Image.Image]) -> list[dict]:
        """
        Return **token-level** dicts, each with a `conf` score so later stages
        can reason about quality.

        ── Behaviour ──────────────────────────────────────────────────────
        • Run Tesseract first (fast).
        • For any word with `conf < 40` (or missing), re-OCR that crop with
          TrOCR.  Every replacement is logged:
              [DBG] pg1:0032  TrOCR '' → 'Walmart'
        • At the end of every page we print a one-line summary like:
              Page 1 → 70 tokens · 59 <50 conf · 55 TrOCR replacements
        • After all pages a magenta table is printed so your pytest -s run
          clearly shows which pages triggered the handwriting model.
        """
        spans: list[dict] = []
        page_summaries: list[str] = []

        for pg_no, img in enumerate(images, start=1):
            tess = pytesseract.image_to_data(
                img, config=TESS_CFG, output_type=pytesseract.Output.DICT
            )

            total_tok, low_conf_tok, trocr_repl = 0, 0, 0

            for i in range(len(tess["level"])):
                txt  = tess["text"][i].strip()
                conf = float(tess["conf"][i])              # -1 or 0-100
                x, y, w, h = (
                    tess["left"][i],  tess["top"][i],
                    tess["width"][i], tess["height"][i],
                )

                total_tok += 1
                if conf < 50:
                    low_conf_tok += 1

                # ── handwriting fallback ────────────────────────────────
                if (not txt or conf < 40) and w > 10 and h > 10:
                    crop = img.crop((x, y, x + w, y + h))
                    txt_before = txt
                    txt  = trocr_text(crop).strip()
                    if txt:
                        print(f"[DBG] pg{pg_no}:{i:04d}  TrOCR '{txt_before}' → '{txt}'")
                        trocr_repl += 1
                        conf = 100.0           # treat TrOCR hits as “high-conf”

                if not txt:
                    continue

                spans.append({
                    "page": pg_no,
                    "text": txt,
                    "x0" : x,          "y0" : y,
                    "x1" : x + w,      "y1" : y + h,
                    "conf": conf,
                })

            page_summaries.append(
                f"Page {pg_no} → {total_tok} tokens · "
                f"{low_conf_tok} <50 conf · {trocr_repl} TrOCR replacements"
            )

        # ── pretty final print ───────────────────────────────────────────
        if page_summaries:
            print("\n\033[95m── OCR summary per page (total / low-conf / trocr) ──\033[0m")
            for line in page_summaries:
                print(line)
            print("──────────────────────────────────────────────────────────────\n")

        return spans

      


    def _merge_tokens(self, typed: list[dict], ocr: list[dict]) -> list[dict]:
        """Keep precise typed spans and add OCR words not already present."""
        merged: list[dict] = typed[:]
        for w in ocr:
            # near-duplicate if same page + centre within 3 px
            dup = next((
                t for t in merged
                if t["page"] == w["page"]
                and abs((t["x0"]+t["x1"])/2 - (w["x0"]+w["x1"])/2) <= 3
                and abs((t["y0"]+t["y1"])/2 - (w["y0"]+w["y1"])/2) <= 3
                and t["text"].casefold() == w["text"].casefold()
            ), None)
            if not dup:
                merged.append(w)
        return merged        




    def _row_chunks(self, rows: list[dict], bytes_per_chunk: int = 30_000):
        """Yield successive slices of `rows` that JSON-encode to ≤bytes_per_chunk."""
        chunk, size = [], 0
        for r in rows:
            s = len(json.dumps(r, ensure_ascii=False))
            if size + s > bytes_per_chunk and chunk:
                yield chunk
                chunk, size = [], 0
            chunk.append(r); size += s
        if chunk:
            yield chunk
    
    def _merge_initial_pages(self, base: dict, new: dict) -> dict:
        out = base.copy()
        for k, v in new.items():
            if isinstance(v, list):
                # simple de‑dupe on tuple‑dump; refine if needed
                seen = {json.dumps(row, sort_keys=True) for row in out.get(k, [])}
                out.setdefault(k, [])
                for row in v:
                    dumped = json.dumps(row, sort_keys=True)
                    if dumped not in seen:
                        out[k].append(row); seen.add(dumped)

            elif isinstance(v, dict):
                out.setdefault(k, {})
                for subk, subv in v.items():
                    if subv is not None:                     # <-- KEEP first good value
                        out[k][subk] = subv or out[k].get(subk)

            else:                                           # scalar
                if v is not None:
                    out[k] = v
        return out

    # ────────────────────────────────────────────────────────────────────────
    # NEW helper that iterates through OCR chunks and enriches bboxes
    # ────────────────────────────────────────────────────────────────────────
    def _enrich_bboxes(
        self,
        initial_json: dict,
        rows: list[dict],
        run_id: str = "o3"
    ) -> dict:

        if not self.llm:
            return initial_json          # ← safety for test-runs w/out LLM

        enriched = initial_json
        out_dir  = pathlib.Path("tests/scrapers"); out_dir.mkdir(exist_ok=True)

        for idx, slice_ in enumerate(self._row_chunks(rows), start=1):
            prompt_text = self.enrich_prompt.format(
                initial_json=json.dumps(enriched, ensure_ascii=False),
                tokens_json=json.dumps(slice_,   ensure_ascii=False),
            )
            msg = HumanMessage(content=[{"type": "text", "text": prompt_text}])

            
            print(msg)
            raw  = self.llm.invoke([msg]).content           # LLM call
            enriched = self._enrich_parser.parse(raw).model_dump()  # 🚦 strict parse
            print("enriched")
            print(enriched)
            # artefact for offline debugging
            (out_dir / f"{run_id}_chunk{idx}.json").write_text(
                json.dumps(enriched, indent=2, ensure_ascii=False)
            )

            # except Exception as exc:
            #     self.log.warning("bbox-enrich chunk %s failed: %s", idx, exc)
            #     continue   # keep the last good version and move on

        return enriched

    
    # ── quick-n-dirty visibility helpers ───────────────────────────────────

    def _initial_to_filing(self, src: dict) -> dict:
        """
        Convert the enriched `initial_json` into the strict Filing shape
        expected downstream:

            • report_meta
            • schedule_a   (monetary contributions)
            • schedule_b   (in-kind contributions)
            • schedule_c   (expenditures / disbursements)
            • schedule_d   (loans / other transactions)

        Notes
        -----
        • Missing scalars are zero-/null-filled.
        • Schedule C still renames `name` → `payee`.
        """
        # Helper to wrap any scalar in the {value, bbox} dict your UI expects
        def bbox_field(val=""):
            return {"value": val, "bbox": None}

        # ── 1. Header / summary meta ───────────────────────────────────
        cand = src.get("candidate", {})
        meta = {
            "candidate_name"                   : bbox_field(cand.get("name", "")),
            "address"                          : bbox_field(cand.get("address")),
            "city"                             : bbox_field(cand.get("city")),
            "state"                            : bbox_field(),
            "zip_code"                         : bbox_field(cand.get("zip")),
            "county"                           : bbox_field(cand.get("county")),
            "office_sought"                    : bbox_field(cand.get("office_sought")),
            "district"                         : bbox_field(cand.get("district")),
            "report_date"                      : bbox_field(),
            "period_start"                     : bbox_field(),
            "period_end"                       : bbox_field(),
            "cash_on_hand_beginning"           : bbox_field(src.get("summary", {})
                                                         .get("cash_on_hand_beginning")),
            "total_contributions_receipts"     : bbox_field(src.get("summary", {})
                                                         .get("total_contributions")),
            "cash_available"                   : bbox_field(src.get("summary", {})
                                                         .get("cash_available")),
            "total_expenditures_disbursements" : bbox_field(src.get("summary", {})
                                                         .get("total_expenditures")),
            "cash_on_hand_closing"             : bbox_field(src.get("summary", {})
                                                         .get("cash_on_hand_close")),
            "signature_name"                   : bbox_field(),
            "signature_date"                   : bbox_field(),
        }

        # ── 2. Schedules ───────────────────────────────────────────────
        sched_a = src.get("contributions", [])             # Schedule A

        sched_b = src.get("in_kind_contributions", [])     # Schedule B  ← NEW

        # Schedule C – rename `name` → `payee`
        sched_c = [
            {
                "date"   : row.get("date"),
                "payee"  : row.get("name"),
                "address": row.get("address"),
                "purpose": row.get("purpose"),
                "amount" : row.get("amount"),
                "bbox"   : row.get("bbox"),
            }
            for row in src.get("expenditures", [])
        ]

        sched_d = src.get("other_transactions", [])        # Schedule D  ← NEW

        # ── 3. Assemble strict Filing dict ─────────────────────────────
        return {
            "report_meta": meta,
            "schedule_a" : sched_a,
            "schedule_b" : sched_b,
            "schedule_c" : sched_c,
            "schedule_d" : sched_d,
        }



    # ── build_filing_payload -------------------------------------------------
    # ── build_filing_payload -------------------------------------------------
    def _build_filing_payload(self, pdf_bytes: bytes) -> dict:
        """
        Convert one Sedgwick PDF to a structured Filing-shaped dict.

        Pipeline
        ------------------------------------------------------------------
        0.  GPT-4o runs **once per page** → `initial_json` whose sub-objects
            carry a `_page` hint.
        1.  Typed-text spans + Tesseract OCR (with `conf`) + optional TrOCR
            fallback → merged tokens → row grouping.
        2.  For every page that appears in `initial_json`, run GPT-4o again
            on *only that page’s rows* to inject `"bbox"` data.
        3.  (Optional) legacy “strict schema” vision prompt (commented out).
        4.  Always return a schema-valid stub from `_fallback_filing`, then
            attach:
                • "_initial_llm_json" – final, bbox-enriched object
                • "tokens"            – grouped rows for debugging
        """

        # helper – nice coloured section headers when run in tty / pytest -s
        def _pfx(tag: str) -> str:
            return f"\n\033[96m── {tag} ────────────────────────────────\033[0m"

        # ── 0 · FIRST-PASS LLM ────────────────────────────────────────────
        print(_pfx("0·LLM first-pass"))
        page_images   = self._pdf_pages_to_images(pdf_bytes)
        print("page imags")
        print(page_images)
        initial_json  = self._initial_llm_extract(page_images)      # may be None
        print("initial_json keys →", list(initial_json or {}))

        # ── 1 · TOKENISATION + OCR  ───────────────────────────────────────
        print(_pfx("1·OCR + typed spans"))
        typed = self._extract_typed_spans(pdf_bytes)
        print(f"typed spans: {len(typed):,}")
        print(typed)            
        ocr_spans = self._extract_ocr_spans(page_images)            # has 'conf'
        print(f"tesseract + handwriting spans: {len(ocr_spans):,}")
        print(ocr_spans)
        # strip the 'conf' key before anything is sent to the LLM
        ocr_final = [
            {k: v for k, v in span.items() if k != "conf"}
            for span in ocr_spans
        ]

        tokens_raw = self._merge_tokens(typed, ocr_final)
        rows       = self._group_rows(tokens_raw)
        print(f"grouped rows: {len(rows):,}")

        # bucket rows by page to drive the second-pass enrichment
        rows_by_pg = defaultdict(list)
        for r in rows:
            rows_by_pg[r["page"]].append(r)

        # 🔸  heuristic pass
        initial_json = self._attach_bboxes_heuristic(initial_json, rows_by_pg)

        # ── 2 · BBOX ENRICHMENT ───────────────────────────────────────────
        if initial_json:
            print(_pfx("2·bbox-enrich"))
            for pg_no, slice_ in rows_by_pg.items():
                print(f"→ page {pg_no} · rows {len(slice_):,}")
                initial_json = self._enrich_bboxes(
                    initial_json=initial_json,
                    rows=slice_,
                    run_id=f"o3p{pg_no}",          # artefacts → o3p<pg>_chunk*.json
                )

        # ── 3 · BUILD FALLBACK STUB  ──────────────────────────────────────
        print(_pfx("3·assemble stub"))
        if not initial_json:
            raise RuntimeError("LLM extraction failed – no initial_json produced")

        filing_dict = self._initial_to_filing(initial_json)
        filing_dict["_initial_llm_json"] = initial_json
        filing_dict["tokens"]            = rows
        print("stub complete ✅")

        # extra log for CI / non-TTY runs (truncate to 400 chars)
        print("final enriched initial_json (trunc):")
        print(json.dumps(initial_json, indent=2, ensure_ascii=False))

        return filing_dict
    


    # ---------- portal helpers ----------------------------------------------
    def _discover_query_id(self) -> str:
        url = f"{self.ROOT}/PublicAccess/api/CustomQuery"
        resp = self._get(url)
        data = resp.json().get("Data", []) if resp else []
        for d in data:
            if d["Name"].startswith("Sedgwick Elections Campaign Expense Reports"):
                return d["ID"]
        raise RuntimeError("Query ID not found – portal changed")

    def _fetch_office_keywords(self, qid: str) -> List[str]:
        url = f"{self.ROOT}/PublicAccess/api/Keywords"
        data = self._post_json(url, {"QueryID": qid}) or {}
        for kw in data.get("Data", []):
            if kw["Name"] == "ELE-Office":
                return kw["Dataset"]
        return []

    def _search_documents(self, qid: str, year: str, office: str):
        url = f"{self.ROOT}/PublicAccess/api/CustomQuery/KeywordSearch"
        payload = {
            "QueryID": int(qid),
            "Keywords": [
                {"ID": 244, "Value": office, "KeywordOperator": "="},
                {"ID": 593, "Value": "",      "KeywordOperator": "="},
                {"ID": 594, "Value": "",      "KeywordOperator": "="},
                {"ID": 727, "Value": year,    "KeywordOperator": "="},
            ],
            "QueryLimit": 0,
        }
        data = self._post_json(url, payload) or {}
        # ⬇︎ keep ONLY names that contain the magic string
        return [d for d in data.get("Data", [])
                if "EXPENDITURE REPORTS" in d["Name"].upper()]


    def _download_pdf(self, doc_id: str):
        url = f"{self.ROOT}/PublicAccess/api/Document/{doc_id}/"
        resp = self._get(url)
        return resp.content if resp else None

        # ---------- public API ---------------------------------------------------
    

    def _slugify(self, txt: str, *, default: str = "unknown") -> str:
        """
        Lower-case, replace every non-[a-z0-9] run with “-”, collapse doubles,
        trim.  Never returns the empty string – falls back to `default`.
        """
        txt = self._SLUG_PAT.sub("-", txt.lower()).strip("-")
        return txt or default


    # ------------------------------------------------------------------
    def _upload_pdf_and_images(
        self,
        pdf_bytes: bytes,
        candidate_slug: str,
        doc_id: str,
    ) -> dict[str, str]:
        """
        Upload the raw PDF + a PNG thumbnail of every page.
        Returns a dict of public gs:// urls you can stash in Firestore.
        """
        urls: dict[str, str] = {}
        today = datetime.datetime.utcnow().strftime("%Y/%m/%d")
        prefix = f"{candidate_slug}/{today}/{doc_id}"

        # PDF ------------------------------------------------------------
        pdf_blob = self.bucket.blob(f"{prefix}.pdf")
        pdf_blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        urls["pdf"] = f"gs://{self.bucket_name}/{pdf_blob.name}"

        # page images ----------------------------------------------------
        try:                                            # don't fail scraping if PIL chokes
            for i, img in enumerate(self._pdf_pages_to_images(pdf_bytes), start=1):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)
                blob = self.bucket.blob(f"{prefix}_page{i}.png")
                blob.upload_from_filename(tmp.name, content_type="image/png")
                urls[f"page_{i}"] = f"gs://{self.bucket_name}/{blob.name}"
        except Exception as exc:
            self.log.warning("PNG upload failed: %s", exc)

        return urls

    # ------------------------------------------------------------------
    def _persist_file_ref(
        self,
        candidate_slug: str,
        doc_id: str,
        office: str,
        year: str,
        metadata: dict,
        file_urls: dict[str, str],
    ) -> None:
        """
        A tiny helper so the front-end can find where each PDF lives.
        """
        self.col_files.document(f"{candidate_slug}__{doc_id}").set(
            {
                "office"   : office,
                "cycle"    : year,
                "source"   : "SEDGWICK",
                "metadata" : metadata,
                "files"    : file_urls,
                "uploaded" : firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    # ------------------------------------------------------------------
    def _normalise_value(self, val):
        """
        Turn every scalar into the standard Firestore doc fragment:
            {value: X, bbox: [...], validated: False}
        """
        if isinstance(val, dict) and {"value", "bbox"} <= val.keys():
            return {
                "value"     : val["value"],
                "bbox"      : val.get("bbox"),
                "validated" : False,
            }
        return {"value": val, "bbox": None, "validated": False}

    # ------------------------------------------------------------------
    def _persist_filing(
        self,
        filing: dict,
        pdf_urls: dict[str, str],
        office: str,
        year: str,
        doc_id: str,
        metadata: dict,
    ) -> None:
        """
        Break the Filing dict into raw rows + roll-ups and push to Firestore.
        Mirrors KentuckyFinanceScraper._commit().
        """
        cand_name   = filing["report_meta"]["candidate_name"]["value"] or "Unknown"
        candidate_slug = self._slugify(cand_name)
        cycle_key   = f"{year}_{'GENERAL' if 'general' in office.lower() else 'UNKNOWN'}"

        batch_raw   = self.db.batch()
        batch_sum   = self.db.batch()
        batch_donor = self.db.batch()

        cand_doc = self.col_summary.document(candidate_slug)

        # ---------- report-meta -----------------------------------------
        summary_seed = {
            "candidate_name" : cand_name,
            "office"         : office,
            "jurisdiction"   : "SEDGWICK_KS",
            "cycles"         : {cycle_key: {}},
            "last_filed_doc" : doc_id,
            "file_urls"      : pdf_urls,
            "updated_at"     : firestore.SERVER_TIMESTAMP,
        }
        batch_sum.set(cand_doc, summary_seed, merge=True)

        # attach every meta scalar (with bbox + validated)
        meta_updates = {}
        for label, obj in filing["report_meta"].items():
            meta_updates[f"cycles.{cycle_key}.meta.{label}"] = self._normalise_value(
                obj
            )
        batch_sum.update(cand_doc, meta_updates)

        # ---------- schedule A (contributions) --------------------------
        for row in filing["schedule_a"]:
            row_id = uuid.uuid4().hex
            is_individual = not any(
                k in row["contributor"].lower()
                for k in (
                    "pac", "inc", "llc", "company", "committee", "bank", "union",
                    "association", "club", "trust",
                )
            )

            data = {
                "candidate_slug"  : candidate_slug,
                "candidate_name"  : cand_name,
                "made_to"         : cand_name,
                "cycle_key"       : cycle_key,
                "jurisdiction"    : "SEDGWICK_KS",
                "filing_source"   : "SEDGWICK",
                "doc_id"          : doc_id,
                "row_type"        : "contribution",
                "contributor"     : self._normalise_value(row["contributor"]),
                "contributor_lc"  : row["contributor"].lower(),
                "address"         : self._normalise_value(row.get("address")),
                "occupation"      : self._normalise_value(row.get("occupation")),
                "amount"          : self._normalise_value(row["amount"]),
                "date"            : self._normalise_value(row["date"]),
                "bbox"            : row.get("bbox"),
                "validated"       : False,
                "scraped_at"      : firestore.SERVER_TIMESTAMP,
            }
            batch_raw.set(self.col_contrib.document(row_id), data)

            # --- roll-ups
            incr = {
                f"cycles.{cycle_key}.last_updated": firestore.SERVER_TIMESTAMP,
                f"cycles.{cycle_key}.raised_total": Increment(row["amount"]),
                f"cycles.{cycle_key}.txns"        : Increment(1),
            }
            if is_individual:
                incr[f"cycles.{cycle_key}.individual_total"] = Increment(row["amount"])
                incr[f"cycles.{cycle_key}.individual_count"] = Increment(1)
            else:
                incr[f"cycles.{cycle_key}.pac_total"] = Increment(row["amount"])
            batch_sum.update(cand_doc, incr)

            # donor roll-up
            donor_slug = self._slugify(row["contributor"])
            don_doc = self.col_donors.document(donor_slug)
            batch_donor.set(
                don_doc,
                {
                    "name"           : row["contributor"],
                    "is_org"         : not is_individual,
                    "jurisdictions"  : firestore.ArrayUnion(["SEDGWICK_KS"]),
                    "total"          : Increment(row["amount"]),
                    "txns"           : Increment(1),
                    "updated_at"     : firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

        # ---------- schedule C (expenditures) ---------------------------
        for row in filing["schedule_c"]:
            row_id = uuid.uuid4().hex
            data = {
                "candidate_slug" : candidate_slug,
                "candidate_name" : cand_name,
                "cycle_key"      : cycle_key,
                "jurisdiction"   : "SEDGWICK_KS",
                "filing_source"  : "SEDGWICK",
                "doc_id"         : doc_id,
                "row_type"       : "expenditure",
                "payee"          : self._normalise_value(row["payee"]),
                "address"        : self._normalise_value(row.get("address")),
                "purpose"        : self._normalise_value(row.get("purpose")),
                "amount"         : self._normalise_value(row["amount"]),
                "date"           : self._normalise_value(row["date"]),
                "bbox"           : row.get("bbox"),
                "validated"      : False,
                "scraped_at"     : firestore.SERVER_TIMESTAMP,
            }
            batch_raw.set(self.col_expend.document(row_id), data)

            # --- roll‑up spending -------------------------------------------------
            incr_exp = {
                f"cycles.{cycle_key}.last_updated": firestore.SERVER_TIMESTAMP,
                f"cycles.{cycle_key}.spent_total" : Increment(row["amount"]),
                f"cycles.{cycle_key}.txns"        : Increment(1),   # duplicates contrib counter but OK
            }
            batch_sum.update(cand_doc, incr_exp)


        # ---------- schedule B (in-kind) ------------------------------
        for row in filing.get("schedule_b", []):
            row_id = uuid.uuid4().hex
            data = {
                "candidate_slug": candidate_slug,
                "candidate_name": cand_name,
                "cycle_key"     : cycle_key,
                "jurisdiction"  : "SEDGWICK_KS",
                "filing_source" : "SEDGWICK",
                "doc_id"        : doc_id,
                "row_type"      : "in_kind",           # ⭐ NEW TYPE
                "contributor"   : self._normalise_value(row["contributor"]),
                "description"   : self._normalise_value(row.get("description")),
                "amount"        : self._normalise_value(row["value"]),
                "date"          : self._normalise_value(row["date"]),
                "bbox"          : row.get("bbox"),
                "validated"     : False,
                "scraped_at"    : firestore.SERVER_TIMESTAMP,
            }
            batch_raw.set(self.col_contrib.document(row_id), data)

    # ---------- schedule D (debts / loans) ------------------------
        for row in filing.get("schedule_d", []):
            row_id = uuid.uuid4().hex
            data = {
                "candidate_slug": candidate_slug,
                "candidate_name": cand_name,
                "cycle_key"     : cycle_key,
                "jurisdiction"  : "SEDGWICK_KS",
                "filing_source" : "SEDGWICK",
                "doc_id"        : doc_id,
                "row_type"      : "debt",              # ⭐ NEW TYPE
                "creditor"      : self._normalise_value(row["creditor"]),
                "purpose"       : self._normalise_value(row.get("purpose")),
                "amount"        : self._normalise_value(row["balance"]),
                "date"          : self._normalise_value(row["date"]),
                "bbox"          : row.get("bbox"),
                "validated"     : False,
                "scraped_at"    : firestore.SERVER_TIMESTAMP,
            }
            batch_raw.set(self.col_expend.document(row_id), data)


        # ---------- commit all three batches ----------------------------
        batch_raw.commit()
        batch_sum.commit()
        batch_donor.commit()

        # ---------- file-reference doc ----------------------------------
        self._persist_file_ref(candidate_slug, doc_id, office, year, metadata, pdf_urls)

    # ------------------------------------------------------------------
    # override the public run() so every filing is persisted automatically
    # ------------------------------------------------------------------
    def run(
        self,
        year: int | str,
        office: str | None = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Works exactly like before **but** each filing is immediately pushed to
        Firestore / Cloud-Storage.  The generator still yields the same package
        so your existing API route / tests remain unchanged.
        """
        year = str(year)
        qid = self._discover_query_id()
        offices = [office] if office else self._fetch_office_keywords(qid)

        for off in offices:
            docs = self._search_documents(qid, year, off)
            for doc in docs:
                doc_id = doc["ID"]
                pdf_bytes = self._download_pdf(doc_id)
                if not pdf_bytes:
                    continue

                # ---- parse --------------------------------------------------
                filing_dict = self._build_filing_payload(pdf_bytes)

                # ---- store files + data ------------------------------------
                urls = self._upload_pdf_and_images(pdf_bytes,
                                                   candidate_slug=self._slugify(
                                                       filing_dict["report_meta"]["candidate_name"]["value"] or "unknown"
                                                   ),
                                                   doc_id=doc_id)
                self._persist_filing(
                    filing=filing_dict,
                    pdf_urls=urls,
                    office=off,
                    year=year,
                    doc_id=doc_id,
                    metadata=doc,
                )

                # ---- yield same payload your API route expects -------------
                yield {
                    "doc_id"   : doc_id,
                    "office"   : off,
                    "metadata" : doc,
                    "pdf_bytes": pdf_bytes,
                    "filing"   : filing_dict,
                }

# ----- CLI tester -------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sedgwick campaign‑finance scraper")
    ap.add_argument("--year", type=int, required=True, help="election cycle, e.g. 2023")
    ap.add_argument("--office", help="exact ELE‑Office string (optional)")
    ap.add_argument("--out", type=pathlib.Path, help="folder where PDFs are stored")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR step for speed")
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    scraper = SedgwickExpenseScraper(ocr=not args.no_ocr)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    for rec in scraper.run(args.year, args.office):
        print(f"{rec['office']} – {rec['metadata'].get('DOC_TYPE'):30} {rec['doc_id']}")
        if args.out:
            fname = args.out / f"{hashlib.sha1(rec['doc_id'].encode()).hexdigest()}.pdf"
            fname.write_bytes(rec["pdf_bytes"])
