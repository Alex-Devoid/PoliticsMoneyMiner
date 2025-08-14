"""
Sedgwick County – Campaign-Finance scraper + processor
⇢ Minimal test harness for the 2023 *Wichita City Mayor* race, or any single
  office you pass on the CLI.  The scraper still supports crawling **all**
  offices when you omit --office.
"""

from __future__ import annotations
import gc
from rich.console   import Console
from rich.progress  import Progress, BarColumn, TimeElapsedColumn
from rich.logging   import RichHandler    # optional
import re, difflib
from collections import defaultdict
# ── stdlib ────────────────────────────────────────────────────────────────────
import io, base64, os, json, logging, pathlib, argparse, hashlib
from json import JSONDecodeError
import uuid, datetime, tempfile  
from typing import Iterator, List, Dict, Any, Optional,TypeAlias, Union
from typing_extensions import TypedDict          # ← Pydantic‑safe TypedDict


import itertools, contextlib
from contextlib import suppress
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
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
import openai
# top‑of‑file imports
from pathlib import Path
import textwrap, uuid, traceback, json, logging, time, io, base64, os
import openai                                    # <‑‑ so we can catch SDK errors

from cloud_function.scrapers.handwriting_ocr import ocr_handwritten
# from cloud_function.scrapers.ocr_backends import trocr_text
from cloud_function.utils.firestore_utils import get_firestore_client
from cloud_function.utils.storage_utils import get_storage_client
from dotenv import load_dotenv 
load_dotenv()
import threading
# we’ll use processes for true CPU parallelism
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp, os
import transformers, logging as _l
transformers.logging.set_verbosity_error()
import traceback
# spawn‑safe on macOS / Windows; fork on Linux
def _init_ocr_worker():
    import os, pytesseract
    os.environ["HF_HOME"] = "/opt/hf_cache"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    global _TROCR_MODEL, _TROCR_PROC
    _TROCR_MODEL  = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    _TROCR_PROC   = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
     

# ------------------------------------------------------------------ #
#  Lazy, process‑safe TrOCR helper (loads once, reused thereafter)   #
# ------------------------------------------------------------------ #
_TROCR_MODEL = _TROCR_PROC = None

# ── OCR runtime switches ──────────────────────────────────────────────
ENABLE_TROCR_FALLBACK = True         # keep handwritten OCR
OCR_WORKERS = 0                      # 0 ⇒ run sequentially (no ProcessPool)
_OCR_POOL   = None                   # pool disabled for dev stability

# Optional: limit Torch thread‑sprawl on CPU
import torch, os
torch.set_num_threads(1)
os.environ.setdefault("HF_HOME", "/opt/hf_cache")          # quiet HUB warnings
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from paddleocr import DocImgOrientationClassification          # PaddleOCR ≥3.0
import numpy as np                                             # PIL→ndarray

# ── NEW: lazy-loaded PaddleOCR orientation helper ────────────
_ORIENT_CLS = None
def _paddle_orientation(img: Image.Image) -> int:
    """
    Return 0/90/180/270 if PaddleOCR’s orientation classifier is confident,
    otherwise 0.  Runs fully on CPU in ≈100 ms.  :contentReference[oaicite:0]{index=0}
    """
    global _ORIENT_CLS
    try:
        if _ORIENT_CLS is None:                      # load once
            _ORIENT_CLS = DocImgOrientationClassification(
                model_name="PP-LCNet_x1_0_doc_ori",  # 4-class model 0–270°
                device="cpu",
                show_log=False,
            )
        # Paddle expects BGR ndarray
        bgr = np.array(img.convert("RGB"))[:, :, ::-1]
        pred = _ORIENT_CLS.predict(bgr, batch_size=1)[0]      # list[Result]
        label = int(pred.res["label_names"][0])               # e.g. '180' → 180
        return label if label in (0, 90, 180, 270) else 0
    except Exception:
        return 0

TESS_TXT = (
    "--psm 6 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

def best_rotation(img: Image.Image, min_letters: int = 25, ratio: float = 1.25) -> int:
    """
    Return 0 / 90 / 180 / 270 if that angle has *significantly* more letters
    than the runner-up; otherwise return 0 (keep original).
    """
    w, h   = img.size
    crop   = img.crop((w*0.15, h*0.15, w*0.85, h*0.85))   # avoid noisy margins
    scores = {}

    for deg in (0, 90, 180, 270):
        txt           = pytesseract.image_to_string(
                           crop.rotate(deg, expand=True), config=TESS_TXT
                         )
        scores[deg]   = sum(c.isalpha() for c in txt)

    best, best_val     = max(scores.items(), key=lambda kv: kv[1])
    second_best_val    = sorted(scores.values(), reverse=True)[1]

    if best_val < min_letters or best_val < second_best_val * ratio:
        return 0                      # vote is weak → leave as-is
    return best

def trocr_text(pil_crop):
    """
    Run microsoft/trocr-small-handwritten on a PIL image crop.

    • Loads the model / processor once, then reuses them.
    • Ensures the crop is RGB so TrOCR doesn’t raise
      “Unsupported number of image dimensions: 2”.
    """
    global _TROCR_MODEL, _TROCR_PROC
    if _TROCR_MODEL is None or _TROCR_PROC is None:
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        _TROCR_PROC  = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        _TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-small-handwritten"
        )
        _TROCR_MODEL.eval()                       # inference mode

    # ✨ guarantee 3‑channel input
    if pil_crop.mode != "RGB":
        pil_crop = pil_crop.convert("RGB")

    inputs = _TROCR_PROC(images=pil_crop, return_tensors="pt")
    ids    = _TROCR_MODEL.generate(
        **{k: v.to("cpu") for k, v in inputs.items()},
        max_new_tokens=24,
    )
    return _TROCR_PROC.batch_decode(ids, skip_special_tokens=True)[0]



# OCR_WORKERS = min(4, (os.cpu_count() or 2))     # bump to #cores ≤ 4

# _OCR_POOL   = ProcessPoolExecutor(
#     max_workers=OCR_WORKERS,
#     mp_context=mp.get_context("spawn"),         # safe on all OSes
#     initializer=_init_ocr_worker,
# )

def _ocr_page(pg_no: int, png_bytes: bytes) -> tuple[list[dict], str]:
    """
    OCR one page (already encoded as PNG bytes) → (token list, summary).
    Runs inside a separate *process* so it must import its own deps.
    """
    import time, pytesseract
    from PIL import Image
    

    TESS_CFG = (
        "--oem 1 --psm 6 "
        "-c tessedit_char_whitelist="
        "0123456789$.,:/-%ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )

    img = Image.open(io.BytesIO(png_bytes))
    spans, total_tok, low_conf, trocr_repl = [], 0, 0, 0
    t_start = time.perf_counter()

    tess = pytesseract.image_to_data(img, config=TESS_CFG,
                                    output_type=pytesseract.Output.DICT)

    for i in range(len(tess["level"])):
        txt  = tess["text"][i].strip()
        conf = float(tess["conf"][i])
        x, y, w, h = (
            tess["left"][i], tess["top"][i],
            tess["width"][i], tess["height"][i],
        )
        total_tok += 1
        if conf < 50:
            low_conf += 1

        # TrOCR fallback for blank / very low‑conf words
        if (conf < 40 or len(txt) < 3) and w > 10 and h > 10:
            guess = trocr_text(img.crop((x, y, x + w, y + h))).strip()
            if guess:
                txt, conf = guess, 100.0
                trocr_repl += 1

        if txt:
            spans.append({
                "page": pg_no, "text": txt,
                "x0": x, "y0": y, "x1": x + w, "y1": y + h,
                "conf": conf,
            })

    summary = (f"Page {pg_no} → {total_tok} tokens · "
            f"{low_conf} <50 conf · {trocr_repl} TrOCR replacements "
            f"({(time.perf_counter()-t_start)*1000:.0f} ms)")
    return spans, summary

# _LOCK is no longer needed – trocr_text() is lru‑cached & thread‑saf


import time, psutil, os
def _mem():                           # quick helper
    return f"{psutil.Process(os.getpid()).memory_info().rss/2**20:6.1f} MB"




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
    @staticmethod
    def _scalar(v):
        """
        Return the plain scalar inside any {value, bbox} wrapper.
        Works for str | int | float | None | FieldBBox‑dict.
        """
        return v.get("value") if isinstance(v, dict) else v

    def _num(self, val) -> float:
        """
        Safely unwrap FieldBBox‑style dicts and coerce the result to float.

        • If *val* is already an int/float → float(val)  
        • If *val* is a {"value": …} wrapper → unwrap first  
        • Anything else (None, "", unparsable) → 0.0
        """
        v = self._scalar(val)       # _scalar() already handles {"value": …}
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0        
    # ── init ───────────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        ocr: bool                = True,
        use_openai: bool         = False,
        # openai_model: str        = "o3",
        openai_model: str        = "gpt-4o",
        
        openai_api_key: str | None = None,
        timeout: int             = 15,
        log_level: int           = logging.DEBUG,
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
            
        # self.col_contrib  = self.db.collection("finance_contributions")
        # self.col_expend   = self.db.collection("finance_expenditures")
        # self.col_summary  = self.db.collection("finance_summary")
        # self.col_donors   = self.db.collection("finance_donors")
        # self.col_files    = self.db.collection("finance_files")   # NEW
        self.col_filings = self.db.collection("filings")
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
        # ── flexible “scalar‑or‑bbox” helpers ─────────────────────────────────────────
        from typing import TypeAlias, Union, List, Dict, Any, Optional
        from pydantic import BaseModel, Field, ConfigDict

        BBox: TypeAlias = List[float]           # convenience
        class FieldBBox(BaseModel):
            value: Union[str, int, float, None]
            bbox: Optional[BBox] = None

        StrOrBox:    TypeAlias = Union[str,  FieldBBox]
        NumberOrBox: TypeAlias = Union[int, float, FieldBBox]

        # ── strict Filing schema (unchanged) ─────────────────────────────────────────
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
            date: StrOrBox
            contributor: StrOrBox
            address: StrOrBox | None = None
            occupation: StrOrBox | None = None
            type: StrOrBox
            amount: NumberOrBox
            bbox: Optional[BBox] = None


        class ScheduleCRow(BaseModel):
            date: StrOrBox
            payee: StrOrBox
            address: StrOrBox | None = None
            purpose: StrOrBox
            amount: NumberOrBox
            bbox: Optional[BBox] = None


        class Filing(BaseModel):
            report_meta: ReportMeta
            schedule_a: List[ScheduleARow]
            schedule_c: List[ScheduleCRow]


        # ── first‑pass extraction schema (wider types) ───────────────────────────────
        class _Cand(BaseModel):
            name: StrOrBox
            address: StrOrBox | None = None
            city: StrOrBox | None = None
            zip: StrOrBox | None = None
            county: StrOrBox | None = None
            office_sought: StrOrBox | None = None
            district: StrOrBox | None = None


        class _Summary(BaseModel):
            cash_on_hand_beginning: NumberOrBox | None = None
            total_contributions:    NumberOrBox | None = None
            cash_available:         NumberOrBox | None = None
            total_expenditures:     NumberOrBox | None = None
            cash_on_hand_close:     NumberOrBox | None = None
            in_kind_contributions:  NumberOrBox | None = None
            other_transactions:     NumberOrBox | None = None


        class _RowBase(BaseModel):
            page: int | None = Field(None, alias="_page")
            model_config = ConfigDict(extra="allow", populate_by_name=True)


        class _Contrib(_RowBase):
            date: StrOrBox
            contributor: StrOrBox
            address: StrOrBox | None = None
            occupation: StrOrBox | None = None
            amount: NumberOrBox


        class _Expend(_RowBase):
            date: StrOrBox
            name: StrOrBox
            address: StrOrBox | None = None
            purpose: StrOrBox | None = None
            amount: NumberOrBox


        class _InKind(_RowBase):
            date: StrOrBox
            contributor: StrOrBox
            description: StrOrBox | None = None
            value: NumberOrBox


        class _Debt(_RowBase):
            date: StrOrBox
            creditor: StrOrBox
            purpose: StrOrBox | None = None
            balance: NumberOrBox


# ── first‑pass extraction schema (wider types) ──────────────────────────
        class _InitialExtract(BaseModel):
            # previously mandatory → now optional
            candidate: _Cand | None = None
            summary:   _Summary | None = None          # <── THIS LINE CHANGED

            contributions:              list[_Contrib] = []
            total_unitemized_contributions: NumberOrBox | None = None
            expenditures:               list[_Expend]  = []
            total_itemized_expenditures: NumberOrBox | None = None
            total_unitemized_expenditures: NumberOrBox | None = None
            total_itemized_receipts:    NumberOrBox | None = None
            other_transactions:         list[_Debt]    = []
            in_kind_contributions:      list[_InKind]  = []

            # accept any stray keys the LLM invents
            model_config = ConfigDict(extra="allow")



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
            date:    StrOrBox
            name:    StrOrBox
            address: StrOrBox | None = None
            purpose: StrOrBox | None = None
            amount:  NumberOrBox
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
{{ "text": "<str>", "page": <int>, "x0": <float>, "y0": <float>, "x1": <float>, "y1": <float> }}
## TASK:
        For every *scalar* value in **initial_json** that appears **in this slice
        only**, find the best-matching token(s) and add a sibling key
"bbox": \[x0, y0, x1, y1]
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
            Coordinates are in native PDF pixels.


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

    
    def _ensure_page_hints(self,
                        initial_json: dict | None,
                        rows_by_pg: dict[int, list[dict]]) -> dict | None:
        """
        Fill the `_page` key for *every* list‑row in `initial_json`
        without relying on the LLM:

        1. Exact bbox → page map
        2. Fallback fuzzy text match
        3. Single‑page shortcut
        """
        if not initial_json:
            return initial_json

        # 1️⃣ bbox → page -------------------------------------------------
        bbox2pg = {
            tuple(r["bbox"]): pg
            for pg, rows in rows_by_pg.items()
            for r in rows
            if r.get("bbox")               # safety – should always be there
        }

        # 2️⃣ quick lowercase text index (one token‑string per row) -------
        txt2pg = {}
        for pg, rows in rows_by_pg.items():
            for r in rows:
                key = " ".join(r["text"].split()).lower()     # collapse whitespace
                if key:                                       # ignore blank lines
                    txt2pg.setdefault(key, pg)

        is_single_page = len(rows_by_pg) == 1
        single_pg_no   = next(iter(rows_by_pg)) if is_single_page else None

        # 3️⃣ apply to every schedule bucket ------------------------------
        for bucket in (
            "contributions",
            "in_kind_contributions",
            "expenditures",
            "other_transactions",
        ):
            for row in initial_json.get(bucket, []):
                if row.get("_page") is not None:
                    continue          # already had a good value

                # (a) bbox match
                pg = bbox2pg.get(tuple(row.get("bbox", [])))

                # (b) fuzzy text match if bbox failed
                if pg is None:
                    # build a terse text signature for the row
                    rough = " ".join(
                        str(row.get(k, "")) for k in (
                            "contributor", "name", "payee", "creditor", "description", "purpose"
                        )
                    ).strip().lower()
                    pg = txt2pg.get(" ".join(rough.split()))

                # (c) single‑page shortcut
                if pg is None and is_single_page:
                    pg = single_pg_no

                # (d) warn if still unknown, but NEVER null/‑1
                if pg is None:
                    self.log.warning("Could not determine page for row: %s", row)
                    pg = single_pg_no or 1        # pragmatic final fallback

                row["_page"] = pg

        return initial_json


    # ── helper: robust page hint extraction ──────────────────────────────
    @staticmethod
    def _page_of(row: dict | None) -> int | None:
        """
        Return whichever page indicator is present in a row
        (`'_page'` preferred, otherwise `'page'`).  If nothing is found,
        returns None.
        """
        if not row:
            return None
        return row.get("_page") or row.get("page")



    def _attach_bboxes_heuristic(
        self,
        initial_json: dict,
        rows_by_pg: dict[int, list[dict]],
        fuzz: float = 0.8,
    ) -> dict:
        if "expenditures" not in initial_json:
            return initial_json

        for exp in initial_json["expenditures"]:
            # skip if already enriched
            if exp.get("bbox"):
                continue

            # ── unwrap new FieldBBox‑style values ──────────────────────────
            name_txt  = self._scalar(exp.get("name")) or ""
            amt_value = self._scalar(exp.get("amount"))

            tgt_name  = re.sub(r"[^A-Za-z0-9]", " ", name_txt).lower()
            tgt_amt   = f"{amt_value:.2f}" if isinstance(amt_value, (int, float)) else None

            best, best_row = 0.0, None
            for pg_rows in rows_by_pg.values():
                for r in pg_rows:
                    txt  = r["text"].lower()
                    score = difflib.SequenceMatcher(None, tgt_name, txt).ratio()
                    if tgt_amt and tgt_amt not in txt:
                        score -= 0.1
                    if score > fuzz and score > best:
                        best, best_row = score, r

            if best_row:
                exp["bbox"]  = best_row["bbox"]
                exp["_page"] = best_row["page"]

        return initial_json

    def _dump_page_debug(
        self,
        *,
        slug: str,
        doc_id: str,
        pg_no: int,
        json_obj: dict | None = None,
        pil_img: Image.Image | None = None,
        pdf_bytes: bytes | None = None,
        stage: str = "rot",              # "orig" | "rot"
    ) -> None:
        """
        Persist debug artefacts under   tests/scrapers/<slug>/<doc_id>/

        • page##_orig.png  – image before rotation
        • page##_rot.png   – image after rotation / just extracted
        • page##_debug.json – the parsed JSON for that page
        • document.pdf     – the original PDF (written once)
        """
        root = Path("tests/scrapers") / slug / doc_id
        root.mkdir(parents=True, exist_ok=True)

        if pil_img is not None:
            pil_img.save(root / f"page{pg_no:02d}_{stage}.png", optimize=True)

        if json_obj is not None:
            (root / f"page{pg_no:02d}_debug.json").write_text(
                json.dumps(json_obj, indent=2, ensure_ascii=False)
            )

        if pdf_bytes and not (root / "document.pdf").exists():
            (root / "document.pdf").write_bytes(pdf_bytes)  



    # ─────────────────────────────────────────────────────────────
    #  Quick OCR context  –  PyMuPDF → Pillow → Tesseract snippet
    # ─────────────────────────────────────────────────────────────
    def _page_ocr_snippet(
        self,
        pdf_bytes: bytes,
        page_no: int,
        *,
        max_chars: int = 600,
    ) -> str:
        """
        Render *page_no* (1-based) to an image, run Tesseract, and return a
        collapsed-whitespace snippet (≤ `max_chars` chars).

        The function keeps its RAM footprint low by:
        • rendering at 150 dpi, which is usually enough for OCR; and
        • calling doc.close() as soon as we have the pixmap.
        """
        import fitz               # PyMuPDF
        import pytesseract, re, io
        from PIL import Image

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if not (1 <= page_no <= len(doc)):
                return ""

            # render just the requested page
            pix = doc[page_no - 1].get_pixmap(dpi=150)     # ← 150 dpi, lighter
            doc.close()                                    # ← RELEASE MEMORY ASAP

            img = Image.open(io.BytesIO(pix.tobytes("png")))
        except Exception:
            # Any failure (bad page index, corrupt PDF, tesseract missing, …)
            # silently falls back to “no OCR context”.
            return ""

        cfg  = (
            "--oem 1 --psm 6 "
            "-c tessedit_char_whitelist=0123456789$.,:/-%ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        text = pytesseract.image_to_string(img, config=cfg)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars] + ("…" if len(text) > max_chars else "")

    

    def needs_flip(self,up_img: Image.Image, min_letters=20) -> bool:
        """Returns True if the 180-deg version looks more like text."""
        txt0 = pytesseract.image_to_string(up_img,   config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        txt180 = pytesseract.image_to_string(
            up_img.rotate(180, expand=True),
            config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        score0   = sum(c.isalpha() for c in txt0)
        score180 = sum(c.isalpha() for c in txt180)
        # Ignore pages that have almost no alpha chars – they won't vote reliably
        if max(score0, score180) < min_letters:
            return False
        return score180 > score0 * 1.15 
    # ------------------------------------------------------------------ #
    #  Cheap CSV logger for OpenAI usage                                 #import 
    # ------------------------------------------------------------------ #
    def _log_openai_usage(
        self,
        *,
        doc_id: str,
        page_no: int,
        stage: str,         # "rotate" or "extract"
        cb,                 # the callback from get_openai_callback()
        csv_path: pathlib.Path = pathlib.Path("openai_usage.csv"),
    ) -> None:
        """
        Append one line of usage data. Creates the file with a header
        the first time it’s called.
        """
        import csv, datetime

        # LangChain ≥0.2 dropped `model_name`; fall back to ""
        model = getattr(cb, "_last_model_name", getattr(cb, "model_name", ""))

        write_header = not csv_path.exists()      # <── this was lost
        with csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)

            if write_header:                      # first time → header line
                writer.writerow(
                    [
                        "timestamp", "doc_id", "page", "stage",
                        "model", "prompt_tokens", "completion_tokens",
                        "total_tokens", "cost_usd",
                    ]
                )

            writer.writerow(
                [
                    datetime.datetime.utcnow().isoformat(timespec="seconds"),
                    doc_id,
                    page_no,
                    stage,
                    model,                        # ← use the safe var
                    cb.prompt_tokens,
                    cb.completion_tokens,
                    cb.total_tokens,
                    cb.total_cost,
                ]
            )

    # -----------------------------------------------------------
    #  LLM-based orientation helpers (stand-alone = easy to mock)
    # -----------------------------------------------------------
    def _llm_rotation_vote(
        self,
        img: Image.Image,
        *,
        llm_rotate,
        rotate_prompt: str,
        returns_current: bool = False,
    ) -> int:
        """
        Ask GPT for the page’s orientation and return **signed CCW** degrees
        (-90 | 0 | 90 | 180).  If anything goes wrong, return 0.

        Setting `returns_current=True` means:
            • the model is asked for its *current* CW angle (0/90/180/270)
            • we translate that to the *needed* CCW correction
            before handing it back to the caller.
        """
        import io, base64
        from langchain_core.messages import HumanMessage

        buf = io.BytesIO()
        img.save(buf, "PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        try:
            # 1. vision call
            raw_cw = llm_rotate.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": rotate_prompt},
                        ]
                    )
                ]
            ).rotate  #          ▲ returns 0 / 90 / 180 / 270 (CW)

            # 2. if the caller asked for the **current** angle, invert it so
            #    downstream gets “how much to rotate CCW to fix the page”.
            raw_needed = (360 - raw_cw) % 360 if returns_current else raw_cw

            # 3. map CW → signed CCW for the pipeline (270 == -90 CW)
            return {0: 0, 90: 90, 180: 180, 270: -90}.get(raw_needed, 0)

        except Exception as exc:
            self.o3_log.warning("LLM rotation vote failed → %s", exc)
            return 0



    def _decide_rotation_deg(
        self,
        img: Image.Image,
        *,
        llm_rotate=None,
        rotate_prompt: str = "",
        returns_current: bool = False,          # NEW – bubbles down to GPT call
    ) -> int:
        """
        Decide page orientation.

        • If `llm_rotate` is supplied → call GPT and **return its answer**.
        • Otherwise → fall back to Paddle OCR, then quick-Tesseract heuristic.

        All angles are **signed CCW degrees**  (-90 | 0 | 90 | 180),
        but helper code is free to pass `returns_current=True` when it wants the
        *current* 0/90/180/270 orientation instead of the needed correction.
        """
        # ── 1 · GPT (only when llm_rotate is provided) ──────────────────────
        # if llm_rotate is not None:
        gpt_deg = self._llm_rotation_vote(
            img,
            llm_rotate=llm_rotate,
            rotate_prompt=rotate_prompt,
            returns_current=returns_current,
        )

        # Log and bail out immediately – we’re “GPT-only” now
        self.o3_log.debug(
            "rotation votes → GPT=%s° (Paddle/Tess skipped)", gpt_deg or "0"
        )
        return gpt_deg    # just in case the model returned 0 / failed

        # # ── 2 · legacy path: Paddle → quick Tesseract ───────────────────────
        # paddle_deg = _paddle_orientation(img)
        # tess_deg   = best_rotation(img) if paddle_deg == 0 else 0

        # self.o3_log.debug(
        #     "rotation votes → GPT=–, Paddle=%s°, Tesseract=%s°",
        #     paddle_deg or "0",
        #     tess_deg   or "0",
        # )
        # return paddle_deg or tess_deg or 0
    # ────────────────────────────────────────────────────────────────────
    #  First‑pass vision LLM (per‑page) – now auto‑detects orientation
    # ────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────
    #  First‑pass vision LLM (per‑page) – function‑calling edition
    # ────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────
    #  First‑pass vision LLM (per‑page) – 100 % structured output
    # ────────────────────────────────────────────────────────────────────

    # ── inside SedgwickExpenseScraper --------------------------------------
    def _upload_page_png(
        self,
        img: Image.Image,
        *,
        candidate_slug: str,
        doc_id: str,
        page_no: int,
    ) -> str:
        """
        Upload one rotated PNG and return its gs:// URL.
        """
        today   = datetime.datetime.utcnow().strftime("%Y/%m/%d")
        prefix  = f"{candidate_slug}/{today}/{doc_id}_page{page_no}.png"

        tmp     = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, optimize=True)

        blob    = self.bucket.blob(prefix)
        blob.upload_from_filename(tmp.name, content_type="image/png")

        return f"gs://{self.bucket_name}/{blob.name}"

    def _initial_llm_extract(
        self,
        page_images: list[Image.Image],
        doc_id: str,
        pdf_bytes: bytes,
    ) -> dict | None:
        """
        1. Ask GPT‑4o if each page image is rotated (‑90 | 0 | 90).
        2. Rotate the PIL image accordingly **before** running the normal
           PAGE_PROMPT extraction.
        3. Parse the page with a second structured‑output call that returns a
           validated `_InitialExtract` object.
        4. Merge every page’s JSON into one dict and tag all sub‑objects
           with `_page`.
        """
        urls_for_firestore: dict[str, str] = {}
        # ── bail‑out if no LLM configured ────────────────────────────────
        if not (self.use_llm and self.o3_llm):
            self.o3_log.debug("LLM disabled – skipping first‑pass extract")
            return None

        # ── (1) rotation helper – function calling ----------------------
        from typing import Literal


#         _ROTATE_PROMPT = """
# **Task:**
# You will receive *one* image containing a single page from a campaign-finance report. Decide whether the page must be rotated so that the text reads naturally **left-to-right, top-to-bottom**.
# **Guidelines for deciding orientation**
# 1. Choose the smallest rotation that makes the page upright:
#    * Already upright → `0`
#    * Rotate 90 ° clockwise → `90`
#    * Rotate 90 ° counter-clockwise → `270`
# 2. Ignore small skews; only consider the three options above.
# **Response format** – return **JSON only**, with no markdown, commentary, or extra keys. Use **exactly** one of:
#
# {"rotate": 0} # {"rotate": 90} # {"rotate": 270} #
# """.strip()    
        _ROTATE_PROMPT = """
You are given one image that contains a single page from a document.

**Task – tell me the page’s current orientation** (clock-wise from upright):

  * Upright →             `0`
  * Rotated 90° CW →      `90`
  * Rotated 180° CW →     `180`
  * Rotated 270° CW →     `270`

Return JSON **only** – no markdown:

{"rotate": 0}
{"rotate": 90}
{"rotate": 180}
{"rotate": 270}
""".strip()    

        class _RotateHint(BaseModel):
            """Return page orientation in degrees CCW."""
            rotate: Literal[-90, 0, 90, 270]

        # Pydantic 2: forward‑ref‑safe
        _RotateHint.model_rebuild()

        llm_rotate = self.o3_llm.with_structured_output(_RotateHint)

        # ── (2) page extractor – function calling + automatic repair ----
        from langchain.output_parsers import (
            OutputFixingParser,
            RetryWithErrorOutputParser,
        )

        # ── (2) page extractor – function calling + automatic repair ──────────
        base_parser   = PydanticOutputParser(
            pydantic_object=self._initial_parser.pydantic_object
        )

        # 1️⃣ auto‑fix pass
        fixing_parser = OutputFixingParser.from_llm(
            llm=self.o3_llm,               # ✅ LLM first (or use keywords)
            parser=base_parser,
        )

        # 2️⃣ retry‑on‑error pass
        retry_parser  = RetryWithErrorOutputParser.from_llm(
            llm=self.o3_llm,               # ✅ LLM first (or use keywords)
            parser=fixing_parser,
        )


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

        **Important Context**
        * Visibile information may be handwritten or typed.
        * Do not treat/convert visible 0s as null. 
        * You are extracting financial data for analysis so it is important to extract numbers precisely, including zeros.
        * I am attaching a basic OCR preview of the page to help you deciper and extract the finacial data

        {self._FORMAT_HINT}

        * Use JSON numbers for money (no `$`, commas, or quotes).  
        * If this page contains none of the above sections, return an empty
          object (`{{}}`).  
        * Output **pure JSON** – **no Markdown**, no commentary.
        """.lstrip()

        merged: dict = {}
        MAX_RETRIES  = int(os.getenv("LLM_PAGE_RETRIES", "5"))

        from rich.console  import Console
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
        )

        console = Console(stderr=True)
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as bar:
            task = bar.add_task(
                "[cyan]LLM extract pages", total=len(page_images)
            )

            # --- helpers --------------------------------------------------------------
            ROT_MAP  = {0: 0, 90: 90, 270: -90, -90: -90, 180: 180}   # GPT → signed CCW
            RAW_DIR  = pathlib.Path("tests/scrapers/raw") / doc_id
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            img_ok, img = False, None

            for pg_no, orig_img in enumerate(page_images, start=1):
                bar.update(task, description=f"[cyan]page {pg_no}/{len(page_images)}")
                time.sleep(1.5)                                   # friendly to rate limits

                # 0️⃣ save the pristine page (only once) -------------------------------
                raw_path = RAW_DIR / f"page{pg_no:02d}_orig.png"
                if not raw_path.exists():
                    orig_img.save(raw_path)

                img_ok, img = False, None          # will flip True when fully processed

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        # ── 1 · original-page artefact (first attempt only) ───────────
                        if attempt == 1:
                            buf = io.BytesIO()
                            orig_img.save(buf, "PNG")
                            self._dump_page_debug(
                                slug="tmp",
                                doc_id=doc_id,
                                pg_no=pg_no,
                                pil_img=orig_img,
                                stage="orig",
                                pdf_bytes=pdf_bytes,
                            )

                        # ── 2 · decide orientation in one helper call ────────────────
                        final_deg = self._decide_rotation_deg(
                            img=orig_img,
                            llm_rotate=llm_rotate,
                            rotate_prompt=_ROTATE_PROMPT,
                            # doc_id=doc_id,
                            # pg_no=pg_no,
                            # pdf_bytes=pdf_bytes,
                            returns_current=True
                        )

                        # ── 3 · apply rotation & cache ───────────────────────────────
                        img = (
                            orig_img.rotate(final_deg, expand=True)
                            if final_deg
                            else orig_img.copy()
                        )
                        page_images[pg_no - 1] = img
                        self.o3_log.info("page %s orientation → %s°", pg_no, final_deg)

                        # ── 4 · build multimodal extraction prompt ──────────────────
                        buf = io.BytesIO()
                        img.save(buf, "PNG")
                        data_url = (
                            "data:image/png;base64,"
                            + base64.b64encode(buf.getvalue()).decode()
                        )
                        ocr_hint = self._page_ocr_snippet(pdf_bytes, pg_no)

                        mm_msg = [
                            {
                                "type": "text",
                                "text": f"### OCR preview (do not re-OCR)\n{ocr_hint}",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": PAGE_PROMPT},
                        ]

                        with get_openai_callback() as cb_ext:
                            raw = self.o3_llm.invoke(
                                [HumanMessage(content=mm_msg)]
                            ).content
                        self._log_openai_usage(
                            doc_id=doc_id,
                            page_no=pg_no,
                            stage="extract",
                            cb=cb_ext,
                        )

                        page_obj = retry_parser.parse_with_prompt(raw, PAGE_PROMPT)
                        page_data = page_obj.model_dump(by_alias=True)

                        # tag every sub-object with _page
                        for k, v in page_data.items():
                            if isinstance(v, dict):
                                v["_page"] = pg_no
                            elif isinstance(v, list):
                                for row in v:
                                    if isinstance(row, dict):
                                        row["_page"] = pg_no

                        merged = self._merge_initial_pages(merged, page_data)

                        # ── 5 · debug artefact (rotated) ────────────────────────────
                        self._dump_page_debug(
                            slug="tmp",
                            doc_id=doc_id,
                            pg_no=pg_no,
                            json_obj=page_data,
                            pil_img=img,
                            stage="rot",
                        )

                        self.o3_log.info(
                            "page %s extracted · cand=%s · A=%s C=%s",
                            pg_no,
                            (page_data.get("candidate") or {}).get("name", "—"),
                            len(page_data.get("contributions", [])),
                            len(page_data.get("expenditures", [])),
                        )

                        bar.advance(task)
                        img_ok = True

                        # ── 6 · upload rotated PNG (after success) ─────────────────
                        cand_slug = self._slugify(
                            (page_data.get("candidate") or {}).get("name", "")
                            or "unknown"
                        )
                        png_url = self._upload_page_png(
                            img,
                            candidate_slug=cand_slug,
                            doc_id=doc_id,
                            page_no=pg_no,
                        )
                        urls_for_firestore[f"page_{pg_no}"] = png_url
                        break  # 🎉 success!

                    except Exception as exc:
                        self.o3_log.error(
                            "page %s attempt %s/%s failed – %s",
                            pg_no,
                            attempt,
                            MAX_RETRIES,
                            exc,
                        )
                        if attempt < MAX_RETRIES:
                            time.sleep(attempt)
                        else:
                            self.o3_log.warning(
                                "page %s skipped after %s attempts",
                                pg_no,
                                MAX_RETRIES,
                            )

                # ── tidy up ─────────────────────────────────────────────────────────
                if img_ok:
                    img.close()
                    orig_img.close()
                    page_images[pg_no - 1] = None
                    gc.collect()
                else:
                    bar.console.print(f"[yellow]page {pg_no} permanently skipped[/]")



                if not img_ok:
                    bar.console.print(f"[yellow]page {pg_no} permanently skipped[/]")


        if not merged:
            self.o3_log.warning(
                "LLM produced no extractable JSON; continuing with heuristics"
            )
            return None

        cand_name = (merged.get("candidate") or {}).get("name", "")
        cand_slug = self._slugify(cand_name) or self._slugify(doc_id)

        # rename any temporary artefact folders that used the placeholder “tmp”
        tmp_root   = pathlib.Path("tests/scrapers/tmp")         # adjust if you used a different path
        final_root = pathlib.Path("tests/scrapers") / cand_slug / doc_id
        if tmp_root.exists():
            final_root.parent.mkdir(parents=True, exist_ok=True)
            tmp_root.rename(final_root)            

        return merged, urls_for_firestore 




    # # ────────────────────────────────────────────────────────────────────
    # #  First-pass vision LLM (per-page) – now with domain hints
    # # ────────────────────────────────────────────────────────────────────
    # def _initial_llm_extract(self, page_images: list[Image.Image],doc_id) -> dict | None:
    #     """
    #     Run GPT‑4o on **each** PDF page.  Every extracted sub‑object gets a `_page`
    #     hint so downstream heuristic matching has context.  If the model fails on
    #     a page we skip it – the rest of the pipeline can still succeed.
    #     """
    #     print("test test test")
    #     if not (self.use_llm and self.o3_llm):
    #         self.o3_log.debug("LLM disabled – skipping first‑pass extract")
    #         print("LLM disabled – skipping first‑pass extract")
    #         return None

    #     merged: dict = {}

    #     # ---------- common text template (re‑used for every page) --------------
    #     PAGE_PROMPT = f"""
    #     You are reading a **Kansas municipal campaign‑finance filing**.  
    #     Each page can contain one (or none) of the following sections:

    #     • **Header / Summary** – candidate details and a 6‑line numeric roll‑up  
    #     • **Schedule A** – “Contributions and Other Receipts”  
    #     • **Schedule B** – “In‑Kind Contributions”  
    #     • **Schedule C** – “Expenditures and Other Disbursements”  
    #     • **Schedule D** – “Other Transactions / Loans / Debts”

    #     Typical column headings you may see:
    #     Date · Name · Address · Occupation · Type · Amount · Purpose · Value ·
    #     Balance · Nature of Account

    #     **TASK**

    #     * Extract **only the information visible on this page**.
    #     * Return a JSON object that conforms to this schema:

    #     **Important Context**
        
    #     * Visibile information may be handwritten or typed.
    #     * The report may be roated 90 degrees counterclockwise or 90 degrees clockwise
    #     * Do not treat/convert visible 0s as null. 
    #     * You are extracting financial data for analysis so it is important to extract numbers precisely, including zeros.
        

    #     {self._FORMAT_HINT}

    #     * Use JSON numbers for money (no `$`, commas, or quotes).  
    #     * If this page contains none of the above sections, return an empty
    #     object (`{{}}`).  
    #     * Output **pure JSON** – **no Markdown**, no commentary.
    #     """
    #     print("start initial extract")
    #     for pg_no, img in enumerate(page_images, start=1):
    #         try:
    #             # ---------- build multimodal message --------------------------
    #             buf = io.BytesIO(); img.save(buf, "PNG")
    #             mm_msg = [
    #                 {"type": "image_url",
    #                 "image_url": {"url": "data:image/png;base64," +
    #                                         base64.b64encode(buf.getvalue()).decode()}},
    #                 {"type": "text", "text": PAGE_PROMPT},
    #             ]

    #             raw = self.o3_llm.invoke([HumanMessage(content=mm_msg)]).content
    #             print(raw)
    #             # Try a strict parse first; fall back to plain json.loads + re‑dump
    #             try:
    #                 page_obj = self._initial_parser.parse(raw)
    #             except Exception:
    #                 page_obj = self._initial_parser.parse(
    #                     json.dumps(json.loads(raw))   # tolerate spacing / newlines
    #                 )

    #             page_data = page_obj.model_dump(by_alias=True)
    #             print("page data")
    #             print(page_data)
    #             # ---------- tag every top‑level dict with page number ----------
    #             for k, v in page_data.items():
    #                 if isinstance(v, dict):
    #                     v["_page"] = pg_no
    #                 elif isinstance(v, list):
    #                     for row in v:
    #                         if isinstance(row, dict):
    #                             row["_page"] = pg_no

    #             merged = self._merge_initial_pages(merged, page_data)

    #             # quick artefact for offline QC
    #             (pathlib.Path("tests/scrapers") /
    #             f"o3_p{pg_no} - jeff .json").write_text(
    #                 json.dumps(page_data, indent=2, ensure_ascii=False)
    #             )

    #             img_path = pathlib.Path("tests/scrapers") / f"o3_p{pg_no} - jeff .png"
    #             img.save(img_path, format="PNG") 

    #         except Exception as exc:
    #             self.o3_log.warning("page %s extract failed (%s) – skipping", pg_no, exc)
    #             print(exc)
    #             continue

    #     # --------- don’t blow up the run if the model gave us nothing ----------
    #     if not merged:
    #         self.o3_log.warning("LLM produced no extractable JSON; continuing with heuristics")
    #         return None

    #     pathlib.Path("tests/scrapers/o3_initial.latest.json").write_text(
    #         json.dumps(merged, indent=2, ensure_ascii=False)
    #     )
    #     return merged



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
        # convert at 200 dpi then down‑sample to 1650 px width (~65 % RAM)
        pages = pdf2image.convert_from_bytes(pdf_bytes, dpi=200)
        proc = []
        for p in pages:
            if p.width > 1650:
                p = p.resize((1650, int(p.height * 1650 / p.width)), Image.BICUBIC)
            gray   = ImageOps.grayscale(p)
            gray   = ImageOps.autocontrast(gray, cutoff=1)
            blur   = gray.filter(ImageFilter.MedianFilter(size=3))
            thresh = blur.point(lambda v: 255 if v > 180 else 0)
            proc.append(thresh)
        return proc

    def _extract_typed_spans(self, pdf_bytes: bytes) -> list[dict]:
        t0 = time.perf_counter()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        spans, n_blocks, n_lines = [], 0, 0

        for pageno, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            n_blocks += len(blocks)
            for block in blocks:
                if block["type"] != 0:                   # images etc.
                    continue
                for line in block["lines"]:
                    n_lines += 1
                    for span in line["spans"]:
                        spans.append({
                            "page": pageno,
                            "text": span["text"],
                            "x0":  span["bbox"][0], "y0": span["bbox"][1],
                            "x1":  span["bbox"][2], "y1": span["bbox"][3],
                        })
        dt = (time.perf_counter() - t0) * 1000
        self.log.debug(
            "[typed] pages=%d  blocks=%d  lines=%d  spans=%d  %.0f ms  %s",
            len(doc), n_blocks, n_lines, len(spans), dt, _mem()
        )
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

    # ─────────────────────────────────────────────────────────────────────────




    # ------------------------------------------------------------------
    #  OCR (Tesseract ➜ optional TrOCR fallback, sequential mode)
    # ------------------------------------------------------------------
    def _extract_ocr_spans(
        self,
        images: list[Image.Image],
        *,
        pool=_OCR_POOL,               # kept for signature compat
    ) -> list[dict]:
        """
        Sequential OCR when OCR_WORKERS == 0.

        Encodes each PIL page to PNG bytes and calls `_ocr_page` inline, so
        you get normal Python stack traces instead of BrokenProcessPool.
        """
        if not images:
            return []

        # Encode PIL pages → PNG bytes
        png_bytes: list[bytes] = []
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            png_bytes.append(buf.getvalue())

        spans_all, summaries = [], {}
        with Progress("[progress.description]{task.description}",
                      BarColumn(), TimeElapsedColumn(),
                      console=Console(stderr=True), transient=True) as bar:
            task = bar.add_task("OCR pages", total=len(images))
            for pg_no, b in enumerate(png_bytes, start=1):
                page_spans, summary = _ocr_page(pg_no, b)   # in‑process
                spans_all.extend(page_spans)
                summaries[pg_no] = summary
                bar.advance(task)

        for pg in sorted(summaries):
            self.log.info(summaries[pg])
        self.log.info("Total OCR tokens: %d", len(spans_all))
        return sorted(spans_all, key=lambda d: (d["page"], d["y0"], d["x0"]))



        


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
    
    # ────────────────────────────────────────────────────────────────────────
    #  Safe merge: page-by-page → single JSON object
    # ────────────────────────────────────────────────────────────────────────
    def _merge_initial_pages(self, base: dict, new: dict) -> dict:
        """
        Combine *new* page JSON into *base*.

        • Lists  → append rows that aren’t exact duplicates.
        • Dicts  → keep the first non-null value for each field, but
                tolerate the situation where the existing value is a
                scalar and the new value is a dict (or vice-versa).
        • Scalars → keep the first non-null scalar that shows up.
        """
        out = base.copy()

        for k, v in new.items():

            # ── 1. lists ───────────────────────────────────────────────
            if isinstance(v, list):
                out.setdefault(k, [])
                seen = {json.dumps(row, sort_keys=True) for row in out[k]}
                for row in v:
                    dumped = json.dumps(row, sort_keys=True)
                    if dumped not in seen:
                        out[k].append(row)
                        seen.add(dumped)

            # ── 2. dicts ───────────────────────────────────────────────
            elif isinstance(v, dict):
                # if the existing value is not a dict, promote it to dict
                if not isinstance(out.get(k), dict):
                    out[k] = {}

                for subk, subv in v.items():
                    if subv is None:
                        continue          # ignore nulls
                    # keep the first non-null value we see
                    if out[k].get(subk) in (None, "", [], {}):
                        out[k][subk] = subv

            # ── 3. scalars ─────────────────────────────────────────────
            else:
                if v is None:
                    continue
                # if we previously stored a dict/list here, it’s richer – keep it
                if isinstance(out.get(k), (dict, list)):
                    continue
                out[k] = out.get(k) or v   # keep the first non-null scalar

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

            
            # print(msg)
            raw  = self.llm.invoke([msg]).content           # LLM call
            enriched = self._enrich_parser.parse(raw).model_dump(by_alias=True)  # 🚦 strict parse
            print("enriched")
            # print(enriched)
            # artefact for offline debugging
            (out_dir / f"{run_id}_chunk{idx}.json").write_text(
                json.dumps(enriched, indent=2, ensure_ascii=False)
            )

            # except Exception as exc:
            #     self.log.warning("bbox-enrich chunk %s failed: %s", idx, exc)
            #     continue   # keep the last good version and move on

        return enriched

    
    # ── quick-n-dirty visibility helpers ───────────────────────────────────

    # ── quick‑n‑dirty visibility helpers ───────────────────────────────────
    def _initial_to_filing(self, src: dict) -> dict:
        """
        Convert the enriched `initial_json` into the strict Filing shape
        expected downstream.

            • report_meta
            • schedule_a   (monetary contributions)
            • schedule_b   (in‑kind contributions)
            • schedule_c   (expenditures / disbursements)
            • schedule_d   (loans / other transactions)

        Notes
        -----
        • FieldBBox objects coming from the LLM have the form
            {"value": <scalar>, "bbox": [...]}
        – we unwrap the scalar but **preserve** the bbox.
        • Schedule C still renames `name` → `payee`.
        """

        # ── helpers --------------------------------------------------------
        def bbox_field(val=""):
            """
            Wrap *val* in the {value, bbox} dict the front‑end expects.

            If *val* is already a FieldBBox‑style dict, keep its bbox
            while un‑nesting the scalar; otherwise, create a fresh wrapper
            with bbox = None.
            """
            if isinstance(val, dict) and {"value", "bbox"} <= val.keys():
                return {"value": val["value"], "bbox": val.get("bbox")}
            return {"value": val, "bbox": None}

        # ------------------------------------------------------------------
        cand = src.get("candidate", {})
        summ = src.get("summary",   {})

        # 1️⃣  Header / summary meta
        meta = {
            "candidate_name"                   : bbox_field(cand.get("name")),
            "address"                          : bbox_field(cand.get("address")),
            "city"                             : bbox_field(cand.get("city")),
            "state"                            : bbox_field(),            # no data on KS form
            "zip_code"                         : bbox_field(cand.get("zip")),
            "county"                           : bbox_field(cand.get("county")),
            "office_sought"                    : bbox_field(cand.get("office_sought")),
            "district"                         : bbox_field(cand.get("district")),
            "report_date"                      : bbox_field(),
            "period_start"                     : bbox_field(),
            "period_end"                       : bbox_field(),
            "cash_on_hand_beginning"           : bbox_field(summ.get("cash_on_hand_beginning")),
            "total_contributions_receipts"     : bbox_field(summ.get("total_contributions")),
            "cash_available"                   : bbox_field(summ.get("cash_available")),
            "total_expenditures_disbursements" : bbox_field(summ.get("total_expenditures")),
            "cash_on_hand_closing"             : bbox_field(summ.get("cash_on_hand_close")),
            "signature_name"                   : bbox_field(),
            "signature_date"                   : bbox_field(),
        }

        # 2️⃣  Schedules (keep existing row objects; just ensure _page)
        def _page_of(row):
            return row.get("_page") or row.get("page")

        sched_a = [{**row, "_page": _page_of(row)} for row in src.get("contributions", [])]
        sched_b = [{**row, "_page": _page_of(row)} for row in src.get("in_kind_contributions", [])]

        sched_c = [
            {
                "date"   : row.get("date"),
                "payee"  : row.get("name"),          # rename
                "address": row.get("address"),
                "purpose": row.get("purpose"),
                "amount" : row.get("amount"),
                "bbox"   : row.get("bbox"),
                "_page"  : _page_of(row),
            }
            for row in src.get("expenditures", [])
        ]

        sched_d = [{**row, "_page": _page_of(row)} for row in src.get("other_transactions", [])]

        # 3️⃣  Bundle into strict Filing dict
        return {
            "report_meta": meta,
            "schedule_a" : sched_a,
            "schedule_b" : sched_b,
            "schedule_c" : sched_c,
            "schedule_d" : sched_d,
        }


    # ── build_filing_payload -------------------------------------------------
    # ── build_filing_payload -------------------------------------------------
    def _build_filing_payload(self, pdf_bytes: bytes,doc_id) -> dict:
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

        # ── 0 · FIRST-PASS LLM  (with tiny on‑disk cache) ────────────────
        print(_pfx("0·LLM first-pass"))

        # 1️⃣  Convert PDF to images once – needed later for OCR anyway
        page_images = self._pdf_pages_to_images(pdf_bytes)

        # 2️⃣  Cache location (one file per Sedgwick doc‑ID)
        cache_dir  = pathlib.Path("tests/scrapers/.llm_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{doc_id}_initial.json"
        page_urls: dict[str, str] = {}
        # 3️⃣  Load or create the cached vision‑LLM output
        if cache_path.exists():
            self.log.debug("[LLM‑cache] hit → %s", cache_path)
            with cache_path.open("r", encoding="utf-8") as fh:
                initial_json = json.load(fh)
        else:
            initial_json, page_urls = self._initial_llm_extract(
                page_images=page_images,
                doc_id=doc_id,
                pdf_bytes=pdf_bytes,              # ← NEW argument
            )
            if initial_json:                                              # write for next run
                with cache_path.open("w", encoding="utf-8") as fh:
                    json.dump(initial_json, fh, ensure_ascii=False, indent=2)
                self.log.debug("[LLM‑cache] wrote %s", cache_path)

        print("initial_json keys →", list(initial_json or {}))     # may be None
        print("initial_json keys →", list(initial_json or {}))

        # ── 1 · TOKENISATION + OCR  ───────────────────────────────────────
        print(_pfx("1·OCR + typed spans"))
        typed       = []                      # ← NEW stub
        ocr_spans   = []                      # ← NEW stub
        tokens_raw  = []                      # ← NEW stub
        rows        = []                      # ← NEW stub
        rows_by_pg  = defaultdict(list)       # ← NEW stub
        # typed = self._extract_typed_spans(pdf_bytes)
        # print(f"typed spans: {len(typed):,}")
        # # print(typed)            
        # ocr_spans = self._extract_ocr_spans(page_images)            # has 'conf'
        # print(f"tesseract + handwriting spans: {len(ocr_spans):,}")
        # # print(ocr_spans)
        # # strip the 'conf' key before anything is sent to the LLM
        # ocr_final = [
        #     {k: v for k, v in span.items() if k != "conf"}
        #     for span in ocr_spans
        # ]

        # tokens_raw = self._merge_tokens(typed, ocr_final)
        # rows       = self._group_rows(tokens_raw)
        # print(f"grouped rows: {len(rows):,}")

        # # bucket rows by page to drive the second-pass enrichment
        # rows_by_pg = defaultdict(list)
        # for r in rows:
        #     rows_by_pg[r["page"]].append(r)

        # # 🔸  heuristic pass
        # initial_json = self._attach_bboxes_heuristic(initial_json, rows_by_pg)
        initial_json = self._ensure_page_hints(initial_json, rows_by_pg)
        # ── SAFETY‑NET: if the LLM ever forgets _page, recover it here ──
        if initial_json:
            for bucket in ("contributions",
                           "in_kind_contributions",
                           "expenditures",
                           "other_transactions"):
                for row in initial_json.get(bucket, []):
                    if row.get("_page") is None and row.get("bbox"):
                        # find which page this bbox belongs to
                        bx = row["bbox"]; pg_guess = None
                        for p, rows in rows_by_pg.items():
                            if any(r["bbox"] == bx for r in rows):
                                pg_guess = p; break
                        row["_page"] = pg_guess or -1   # -1 = unknown but not null        

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

        return filing_dict, page_images, page_urls
    


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
                {"ID": 593, "Value": "Blubaugh",      "KeywordOperator": "="},
                {"ID": 594, "Value": "Jeff",      "KeywordOperator": "="},
                
                # {"ID": 593, "Value": "",      "KeywordOperator": "="},
                # {"ID": 594, "Value": "",      "KeywordOperator": "="},
                

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
    

    # ------------------------------------------------------------------
    def _slugify(self, txt: str | dict | None, *, default: str = "unknown") -> str:
        """
        Lower‑case *txt*, replace every non‑[a‑z0‑9] run with “‑”, collapse doubles,
        trim.  Accepts either a raw string **or** the {"value": ..., "bbox": [...]}
        objects produced by the bbox‑enrichment step.
        """
        if isinstance(txt, dict):          # unwrap {"value": ...}
            txt = txt.get("value", "")     # fall back to empty string
        if not isinstance(txt, str):
            txt = str(txt or "")
        txt = self._SLUG_PAT.sub("-", txt.lower()).strip("-")
        return txt or default



    # ------------------------------------------------------------------
    def _upload_pdf_and_images(
        self,
        pdf_bytes: bytes,
        candidate_slug: str,
        doc_id: str,
        page_images: list[Image.Image] | None = None
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

        if page_images is None:                     # backward-compat
            page_images = self._pdf_pages_to_images(pdf_bytes)
        try:                                            # don't fail scraping if PIL chokes
            for i, img in enumerate(page_images, start=1):
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
    # ------------------------------------------------------------------
    #  NEW simplified Firestore persistence
    # ------------------------------------------------------------------
    def _persist_filing(
        self,
        filing  : dict,
        pdf_urls: dict[str, str],
        office  : str,
        year    : str,
        doc_id  : str,
        metadata: dict,
    ) -> None:
        """
        Persist the parsed filing into the new structure:

            filings/<doc_id>
                meta + files
                pages/<n>
                    img
                    rows/<autoId>
        """

        # ── root‑level document -------------------------------------------------
        cand_name      = filing["report_meta"]["candidate_name"]["value"] or "Unknown"
        candidate_slug = self._slugify(cand_name)

        filing_doc = self.col_filings.document(doc_id)
        filing_doc.set(
            {
                "candidate_name" : cand_name,
                "candidate_slug" : candidate_slug,
                "office"         : office,
                "cycle"          : str(year),
                "jurisdiction"   : "SEDGWICK_KS",
                "source"         : "SEDGWICK",
                "meta"           : filing["report_meta"],   # full header / totals block
                "files"          : pdf_urls,                # page_1 … page_N, pdf
                "metadata"       : metadata,                # raw portal JSON
                "scraped_at"     : firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        # ── fan‑out rows by page -----------------------------------------------
        from collections import defaultdict
        page_rows: dict[int, list[dict]] = defaultdict(list)


        # ---- header / summary as QC-rows -------------------------------
        meta_rows = []
        for label, fb in filing["report_meta"].items():          # FieldBBox objects
            meta_rows.append({
                "field"     : label,
                "value"     : fb.get("value"),
                "bbox"      : fb.get("bbox"),
                "_page"     : 1,             # always first page on KS form
                "row_type"  : "meta",
                "validated" : False,
                "row_order" : len(meta_rows) + 1,   # preserve order shown on form
            })
        page_rows[1].extend(meta_rows)        

        def _add(rows: list[dict], row_type: str):
            for idx, r in enumerate(rows, start=1):          # <- keep index
                pg = r.get("_page") or 1
                page_rows[pg].append({
                    **r,
                    "row_type"  : row_type,
                    "validated" : False,
                    "row_order" : idx,        # NEW – 1-based top→bottom
                })
        _add(filing["schedule_a"],              "contribution")
        _add(filing.get("schedule_b", []),      "in_kind")
        _add(filing["schedule_c"],              "expenditure")
        _add(filing.get("schedule_d", []),      "debt")

        # ── batch‑write pages + rows -------------------------------------------
        batch = self.db.batch()

        for pg, rows in page_rows.items():
            page_ref = filing_doc.collection("pages").document(str(pg))

            # page document (PNG url + page number)
            batch.set(
                page_ref,
                {
                    "img"     : pdf_urls.get(f"page_{pg}"),
                    "page_no" : pg,
                    "updated" : firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

            # all rows for this page
            for r in rows:
                batch.set(page_ref.collection("rows").document(), r)

        batch.commit()


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

                # Parse → get rotated-PNG URLs back from first-pass LLM
                filing_dict, page_images, page_urls = self._build_filing_payload(
                    pdf_bytes, doc_id
                )

                # Upload *PDF only* (empty list skips duplicate PNG upload)
                pdf_only_urls = self._upload_pdf_and_images(
                    pdf_bytes,
                    candidate_slug=self._slugify(
                        filing_dict["report_meta"]["candidate_name"]["value"] or "unknown"
                    ),
                    doc_id=doc_id,
                    page_images=[],        # ← prevents second PNG upload
                )

                # merge in the rotated-page PNGs we already uploaded
                pdf_only_urls.update(page_urls)

                # Now persist the whole package
                self._persist_filing(
                    filing=filing_dict,
                    pdf_urls=pdf_only_urls,
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
        level=logging.DEBUG,
    )

    scraper = SedgwickExpenseScraper(ocr=not args.no_ocr)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    for rec in scraper.run(args.year, args.office):
        print(f"{rec['office']} – {rec['metadata'].get('DOC_TYPE'):30} {rec['doc_id']}")
        if args.out:
            fname = args.out / f"{hashlib.sha1(rec['doc_id'].encode()).hexdigest()}.pdf"
            fname.write_bytes(rec["pdf_bytes"])