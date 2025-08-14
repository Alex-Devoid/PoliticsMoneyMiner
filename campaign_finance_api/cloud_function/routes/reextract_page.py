# cloud_function/routes/reextract_page.py
from __future__ import annotations

import io
import os
import json
from datetime import timedelta
from typing import Optional, Tuple, List, Dict, Any, Union, TypeAlias
import base64
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from google.cloud import firestore
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from PIL import Image

from cloud_function.utils.firestore_utils import get_firestore_client
from cloud_function.utils.storage_utils import get_storage_client

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser, RetryWithErrorOutputParser

router = APIRouter(prefix="/finance", tags=["reextract"])

def _png_bytes_to_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()

# ───────────────────────── helpers ─────────────────────────

def _parse_gs_url(gs_url: str) -> Tuple[str, str]:
    if not gs_url.startswith("gs://"):
        raise HTTPException(400, "Expected gs:// storage URL")
    path = gs_url[len("gs://"):]
    bucket, _, blob = path.partition("/")
    if not bucket or not blob:
        raise HTTPException(500, "Malformed gs:// URL")
    return bucket, blob


def _rotate_png_bytes(png_bytes: bytes, deg_cw: int) -> bytes:
    """Rotate CW by 0/90/180/270 and return PNG bytes."""
    deg_ccw = (360 - (deg_cw % 360)) % 360
    im = Image.open(io.BytesIO(png_bytes))
    rot = im.rotate(deg_ccw, expand=True)
    buf = io.BytesIO()
    rot.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _generate_signed_url_for_read(blob, minutes: int = 10) -> str:
    """Signed URL (v4) so OpenAI can read the image directly."""
    try:
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=minutes),
            method="GET",
            response_disposition="inline",
            content_type="image/png",
        )
    except Exception as e:
        raise HTTPException(500, f"Could not create signed URL for extraction: {e}")


# ───────────────────── Pydantic models (match scraper) ─────────────────────

BBox: TypeAlias = List[float]

class FieldBBox(BaseModel):
    value: Union[str, int, float, None]
    bbox: Optional[BBox] = None

StrOrBox:    TypeAlias = Union[str, FieldBBox]
NumberOrBox: TypeAlias = Union[int, float, FieldBBox]

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
    type: StrOrBox | None = None
    amount: NumberOrBox

class _Expend(_RowBase):
    date: StrOrBox
    name: StrOrBox
    address: StrOrBox | None = None
    purpose: StrOrBox | None = None
    amount: NumberOrBox
    bbox: Optional[BBox] = None

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

class _InitialExtract(BaseModel):
    candidate: _Cand | None = None
    summary:   _Summary | None = None
    contributions: list[_Contrib] = []
    expenditures:  list[_Expend]  = []
    in_kind_contributions: list[_InKind] = []
    other_transactions:    list[_Debt]   = []
    # allow anything extra without exploding
    model_config = ConfigDict(extra="allow")


def _unwrap_scalar(v: Any) -> Any:
    """Match scraper: unwrap FieldBBox → value; everything else passthrough."""
    if isinstance(v, dict) and "value" in v:
        return v.get("value")
    return v

def _bbox_field(val: Any = "") -> Dict[str, Any]:
    """Wrap value into {value, bbox} shape used by meta rows."""
    if isinstance(val, dict) and {"value", "bbox"} <= val.keys():
        return {"value": val.get("value"), "bbox": val.get("bbox")}
    return {"value": val, "bbox": None}

def _page_of(row: Dict[str, Any] | None) -> int | None:
    if not row:
        return None
    return row.get("_page") or row.get("page")

def _initial_to_filing(src: Dict[str, Any], *, page_no: int) -> Dict[str, Any]:
    """
    Convert _InitialExtract-like dict → strict Filing dict (same as scraper),
    but only for a single page. Meta rows are attributed to page 1 (scraper behavior).
    """
    cand = src.get("candidate", {}) or {}
    summ = src.get("summary",   {}) or {}

    report_meta = {
        "candidate_name"                   : _bbox_field(cand.get("name")),
        "address"                          : _bbox_field(cand.get("address")),
        "city"                             : _bbox_field(cand.get("city")),
        "state"                            : _bbox_field(""),
        "zip_code"                         : _bbox_field(cand.get("zip")),
        "county"                           : _bbox_field(cand.get("county")),
        "office_sought"                    : _bbox_field(cand.get("office_sought")),
        "district"                         : _bbox_field(cand.get("district")),
        "report_date"                      : _bbox_field(""),
        "period_start"                     : _bbox_field(""),
        "period_end"                       : _bbox_field(""),
        "cash_on_hand_beginning"           : _bbox_field(summ.get("cash_on_hand_beginning")),
        "total_contributions_receipts"     : _bbox_field(summ.get("total_contributions")),
        "cash_available"                   : _bbox_field(summ.get("cash_available")),
        "total_expenditures_disbursements" : _bbox_field(summ.get("total_expenditures")),
        "cash_on_hand_closing"             : _bbox_field(summ.get("cash_on_hand_close")),
        "signature_name"                   : _bbox_field(""),
        "signature_date"                   : _bbox_field(""),
    }

    # Normalize schedule rows to include _page and keep field names identical to scraper
    def _ensure_page(r: Dict[str, Any]) -> Dict[str, Any]:
        if r is None:
            return {}
        # prefer explicit _page; fall back to provided alias 'page' or current page
        pg = r.get("_page") or r.get("page") or page_no
        out = {**r}
        out.pop("page", None)
        out["_page"] = pg
        return out

    schedule_a = [_ensure_page(r) for r in src.get("contributions", [])]
    schedule_b = [_ensure_page(r) for r in src.get("in_kind_contributions", [])]
    # Keep 'name' (NOT 'payee') to match scraper
    schedule_c = [
        _ensure_page({
            "date"   : r.get("date"),
            "name"   : r.get("name") or r.get("payee"),  # accept either, persist as 'name'
            "address": r.get("address"),
            "purpose": r.get("purpose"),
            "amount" : r.get("amount"),
            "bbox"   : r.get("bbox"),
            "_page"  : _page_of(r) or page_no,
        })
        for r in src.get("expenditures", [])
    ]
    schedule_d = [_ensure_page(r) for r in src.get("other_transactions", [])]

    return {
        "report_meta": report_meta,
        "schedule_a" : schedule_a,
        "schedule_b" : schedule_b,
        "schedule_c" : schedule_c,
        "schedule_d" : schedule_d,
    }


# ───────────────────────── models ─────────────────────────

class ReextractRequest(BaseModel):
    rotation: int = Field(..., description="CW degrees: 0, 90, 180, 270")
    # Option A
    doc_id: Optional[str] = None
    local_page: Optional[int] = Field(None, ge=1)
    # Option B
    slug: Optional[str] = None
    global_page: Optional[int] = Field(None, ge=1)
    overwrite_image: bool = True
    reset_validated: bool = True

class ReextractResult(BaseModel):
    doc_id: str
    page: int
    rows_written: int
    gs_url: str
    rotation_applied: int


# ─────────────────── page resolver ────────────────────

async def _resolve_doc_and_local_page(
    client: firestore.Client,
    *,
    doc_id: Optional[str],
    local_page: Optional[int],
    slug: Optional[str],
    global_page: Optional[int],
) -> Tuple[str, int]:
    """Accept (doc_id+local_page) or (slug+global_page) → (doc_id, local_page)."""
    if doc_id and local_page:
        return doc_id, int(local_page)

    if not (slug and global_page):
        raise HTTPException(400, "Provide (doc_id & local_page) or (slug & global_page)")

    filings = (
        client.collection("filings")
              .where("candidate_slug", "==", slug)
              .order_by("scraped_at", direction=firestore.Query.DESCENDING)
              .stream()
    )
    filings = [f for f in filings]
    if not filings:
        raise HTTPException(404, "Candidate not found")

    cursor = 0
    for f in filings:
        d = f.to_dict() or {}
        files = d.get("files", {})
        pages_here = sorted(int(k.split("_")[1]) for k in files if k.startswith("page_"))
        for p in pages_here:
            cursor += 1
            if cursor == int(global_page):
                return f.id, p

    raise HTTPException(404, "global_page out of range for this slug")


# ─────────────── LLM single-page extract (mirrors scraper) ───────────────

async def _extract_initial_for_page(*, image_url: str) -> Dict[str, Any]:
    """
    Run the same structured extraction the scraper uses (page-level),
    returning a dict matching _InitialExtract.
    """
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY is not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0)

    # Parser + fix/retry chain (same style as scraper)
    base_parser   = PydanticOutputParser(pydantic_object=_InitialExtract)
    fixing_parser = OutputFixingParser.from_llm(llm=llm, parser=base_parser)
    retry_parser  = RetryWithErrorOutputParser.from_llm(llm=llm, parser=fixing_parser)

    format_instructions = base_parser.get_format_instructions()
    PAGE_PROMPT = f"""
You are reading a Kansas municipal campaign-finance filing page. It may contain:
  • Header/Summary  • Schedule A (Contributions)  • Schedule B (In-Kind)
  • Schedule C (Expenditures)  • Schedule D (Other Transactions)

TASK:
  • Extract ONLY information visible on this page.
  • Return a JSON object that conforms to this schema:
{format_instructions}

Rules:
  • Use JSON numbers for money (no $, commas, or quotes).
  • If this page contains none of the above sections, return {{}}.
  • Output pure JSON, no markdown or commentary.
""".strip()

    msg = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": PAGE_PROMPT},
        ]
    )

    raw = (await llm.ainvoke([msg])).content
    try:
        page_obj = retry_parser.parse_with_prompt(raw, PAGE_PROMPT)
    except Exception:
        # last-ditch tolerance for slight formatting issues
        page_obj = base_parser.parse(json.dumps(json.loads(str(raw))))

    return page_obj.model_dump(by_alias=True)


# ───────────────────── Firestore write (match scraper) ─────────────────────

def _build_page_rows_for_write(
    filing_dict: Dict[str, Any],
    *,
    page_no: int,
    reset_validated: bool
) -> List[Dict[str, Any]]:
    """
    Build the list of row docs to write for ONE page, mirroring _persist_filing.

    Key compat points with the scraper:
      • meta rows live ONLY on page 1 and always carry _page: 1
      • schedule_c uses 'name' (not 'payee')
      • rows carry '_page' (never the alias 'page')
      • we keep row_order as the 1-based index within each schedule list
    """
    rows: List[Dict[str, Any]] = []

    # 1) Meta rows as QC lines — ONLY write when updating page 1
    if page_no == 1:
        meta_rows = []
        for label, fb in (filing_dict.get("report_meta") or {}).items():
            meta_rows.append({
                "field"     : label,
                "value"     : fb.get("value"),
                "bbox"      : fb.get("bbox"),
                "_page"     : 1,               # KS form places header/totals on page 1
                "row_type"  : "meta",
                "validated" : False if reset_validated else False,
                "row_order" : len(meta_rows) + 1,
            })
        rows.extend(meta_rows)

    def _canon_page(r: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
        """Return (row_without_page_alias, resolved_page)."""
        out = dict(r)
        pg = out.get("_page") or out.get("page") or page_no
        out.pop("page", None)
        out["_page"] = pg
        return out, int(pg)

    # Helper to append schedule rows with types
    def _add(seq: List[Dict[str, Any]], row_type: str):
        for idx, raw in enumerate(seq, start=1):  # keep global index per schedule
            r = dict(raw)
            # schedule_c compat: ensure 'name' key
            if row_type == "expenditure":
                if "payee" in r and "name" not in r:
                    r["name"] = r.pop("payee")

            r, pg = _canon_page(r)
            if pg != page_no:
                continue  # only write rows that belong to this page

            row = {
                **r,
                "row_type"  : row_type,
                "validated" : False if reset_validated else False,
                "row_order" : idx,
            }
            rows.append(row)

    _add(filing_dict.get("schedule_a", []), "contribution")
    _add(filing_dict.get("schedule_b", []), "in_kind")
    _add(filing_dict.get("schedule_c", []), "expenditure")
    _add(filing_dict.get("schedule_d", []), "debt")

    return rows


# ───────────────────── endpoint ─────────────────────

@router.post("/reextract/rotate-extract", response_model=ReextractResult)
async def rotate_and_reextract(
    body: ReextractRequest = Body(...),
    client: firestore.Client = Depends(get_firestore_client),
):
    """
    Rotate one stored PNG page, then re-extract using the same
    LangChain/OpenAI-first-pass JSON shape as the Sedgwick scraper.
    Writes page rows in the same format as the scraper helpers.
    Falls back to a data: URL if we cannot sign a GCS URL locally.
    """
    if body.rotation not in (0, 90, 180, 270):
        raise HTTPException(400, "rotation must be one of 0, 90, 180, 270")

    # 1) Resolve doc_id & local page
    doc_id, local_page = await _resolve_doc_and_local_page(
        client,
        doc_id=body.doc_id,
        local_page=body.local_page,
        slug=body.slug,
        global_page=body.global_page,
    )

    filing_ref = client.collection("filings").document(doc_id)
    filing_doc = filing_ref.get().to_dict() or {}
    files_map = filing_doc.get("files", {})
    gs_url = files_map.get(f"page_{local_page}")
    if not gs_url:
        raise HTTPException(404, f"Missing files.page_{local_page}")

    # 2) Download → rotate → (optional) overwrite image bytes
    storage = get_storage_client()
    bucket_name, blob_path = _parse_gs_url(gs_url)
    blob = storage.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        raise HTTPException(404, "Storage object not found for this page")

    orig_png = blob.download_as_bytes()
    rotated = _rotate_png_bytes(orig_png, body.rotation)

    if body.overwrite_image:
        blob.upload_from_string(rotated, content_type="image/png")

    # 2b) Prepare image reference for the LLM:
    #     try a signed URL first; if credentials can't sign, fall back to a data: URL.
    try:
        image_ref = _generate_signed_url_for_read(blob, minutes=10)
    except Exception:
        image_ref = "data:image/png;base64," + base64.b64encode(rotated).decode()

    # 3) LLM extract (same schema as scraper’s first pass)
    try:
        initial = await _extract_initial_for_page(image_url=image_ref)
    except Exception as e:
        raise HTTPException(500, f"LLM extract failed: {e}")

    if not isinstance(initial, dict):
        initial = {}

    # Ensure each schedule row carries _page (drop any alias later during build)
    for key, val in list(initial.items()):
        if isinstance(val, list):
            for r in val:
                if isinstance(r, dict):
                    r.setdefault("_page", local_page)

    # 4) Convert to strict filing shape (like scraper), then build page rows
    try:
        filing_like = _initial_to_filing(initial, page_no=local_page)
        new_rows = _build_page_rows_for_write(
            filing_like, page_no=local_page, reset_validated=body.reset_validated
        )
    except Exception as e:
        raise HTTPException(500, f"Post-process to page rows failed: {e}")

    # 5) Overwrite ONLY this page's rows in Firestore
    page_ref = filing_ref.collection("pages").document(str(local_page))
    batch = client.batch()

    # wipe existing rows for this page
    for snap in page_ref.collection("rows").stream():
        batch.delete(snap.reference)

    # write new rows for this page
    for r in new_rows:
        batch.set(page_ref.collection("rows").document(), r)

    # bump page doc
    batch.set(
        page_ref,
        {
            "img": gs_url,  # same path; bytes may be rotated
            "page_no": local_page,
            "rotation_applied": body.rotation,
            "updated": SERVER_TIMESTAMP,
        },
        merge=True,
    )

    batch.commit()

    return ReextractResult(
        doc_id=doc_id,
        page=local_page,
        rows_written=len(new_rows),
        gs_url=gs_url,
        rotation_applied=body.rotation,
    )


# Back-compat alias (front-end earlier posted to /reextract)
@router.post("/reextract", response_model=ReextractResult)
async def reextract_alias(
    body: ReextractRequest = Body(...),
    client: firestore.Client = Depends(get_firestore_client),
):
    return await rotate_and_reextract(body, client)
