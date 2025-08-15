# cloud_function/routes/reextract_page.py
from __future__ import annotations

import io
import os
import json
import base64
from datetime import timedelta
from typing import Optional, Tuple, List, Dict, Any, Union, TypeAlias

from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks
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

# at top of file
import logging, time
logger = logging.getLogger("reextract")
logger.setLevel(logging.INFO)

# SINGLE router (keep tags)
router = APIRouter(prefix="/finance", tags=["reextract"])

# ───────────────────────── helpers ─────────────────────────

def _png_bytes_to_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()

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
    model_config = ConfigDict(extra="allow")

def _bbox_field(val: Any = "") -> Dict[str, Any]:
    if isinstance(val, dict) and {"value", "bbox"} <= val.keys():
        return {"value": val.get("value"), "bbox": val.get("bbox")}
    return {"value": val, "bbox": None}

def _page_of(row: Dict[str, Any] | None) -> int | None:
    if not row:
        return None
    return row.get("_page") or row.get("page")

def _initial_to_filing(src: Dict[str, Any], *, page_no: int) -> Dict[str, Any]:
    """Convert _InitialExtract-like dict → strict Filing dict (same as scraper), for single page."""
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

    def _ensure_page(r: Dict[str, Any]) -> Dict[str, Any]:
        if r is None: return {}
        pg = r.get("_page") or r.get("page") or page_no
        out = {**r}
        out.pop("page", None)
        out["_page"] = pg
        return out

    schedule_a = [_ensure_page(r) for r in src.get("contributions", [])]
    schedule_b = [_ensure_page(r) for r in src.get("in_kind_contributions", [])]
    schedule_c = [
        _ensure_page({
            "date": r.get("date"),
            "name": r.get("name") or r.get("payee"),
            "address": r.get("address"),
            "purpose": r.get("purpose"),
            "amount": r.get("amount"),
            "bbox": r.get("bbox"),
            "_page": _page_of(r) or page_no,
        }) for r in src.get("expenditures", [])
    ]
    schedule_d = [_ensure_page(r) for r in src.get("other_transactions", [])]

    return {
        "report_meta": report_meta,
        "schedule_a" : schedule_a,
        "schedule_b" : schedule_b,
        "schedule_c" : schedule_c,
        "schedule_d" : schedule_d,
    }

def _build_page_rows_for_write(
    filing_dict: Dict[str, Any], *, page_no: int, reset_validated: bool
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if page_no == 1:
        meta_rows = []
        for label, fb in (filing_dict.get("report_meta") or {}).items():
            meta_rows.append({
                "field"     : label,
                "value"     : fb.get("value"),
                "bbox"      : fb.get("bbox"),
                "_page"     : 1,
                "row_type"  : "meta",
                "validated" : False if reset_validated else False,
                "row_order" : len(meta_rows) + 1,
            })
        rows.extend(meta_rows)

    def _canon_page(r: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
        out = dict(r)
        pg = out.get("_page") or out.get("page") or page_no
        out.pop("page", None)
        out["_page"] = pg
        return out, int(pg)

    def _add(seq: List[Dict[str, Any]], row_type: str):
        for idx, raw in enumerate(seq, start=1):
            r = dict(raw)
            if row_type == "expenditure" and "payee" in r and "name" not in r:
                r["name"] = r.pop("payee")
            r, pg = _canon_page(r)
            if pg != page_no:
                continue
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

# ─────────────── Resolver, LLM extract (reused) ───────────────

async def _resolve_doc_and_local_page(
    client: firestore.Client,
    *, doc_id: Optional[str], local_page: Optional[int], slug: Optional[str], global_page: Optional[int],
) -> Tuple[str, int]:
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

async def _extract_initial_for_page(*, image_url: str) -> Dict[str, Any]:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY is not set")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0)

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

    msg = HumanMessage(content=[{"type":"image_url","image_url":{"url": image_url}}, {"type":"text","text": PAGE_PROMPT}])
    raw = (await llm.ainvoke([msg])).content
    try:
        page_obj = retry_parser.parse_with_prompt(raw, PAGE_PROMPT)
    except Exception:
        page_obj = base_parser.parse(json.dumps(json.loads(str(raw))))
    return page_obj.model_dump(by_alias=True)

# ───────────────────── Slow endpoint (kept) ─────────────────────

class ReextractRequest(BaseModel):
    rotation: int = Field(..., description="CW degrees: 0, 90, 180, 270")
    doc_id: Optional[str] = None
    local_page: Optional[int] = Field(None, ge=1)
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

@router.post("/reextract/rotate-extract", response_model=ReextractResult)
async def rotate_and_reextract(
    body: ReextractRequest = Body(...),
    client: firestore.Client = Depends(get_firestore_client),
):
    if body.rotation not in (0, 90, 180, 270):
        raise HTTPException(400, "rotation must be one of 0, 90, 180, 270")

    doc_id, local_page = await _resolve_doc_and_local_page(
        client, doc_id=body.doc_id, local_page=body.local_page, slug=body.slug, global_page=body.global_page,
    )

    filing_ref = client.collection("filings").document(doc_id)
    filing_doc = filing_ref.get().to_dict() or {}
    files_map = filing_doc.get("files", {})
    gs_url = files_map.get(f"page_{local_page}")
    if not gs_url:
        raise HTTPException(404, f"Missing files.page_{local_page}")

    storage = get_storage_client()
    bucket_name, blob_path = _parse_gs_url(gs_url)
    blob = storage.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        raise HTTPException(404, "Storage object not found for this page")

    orig_png = blob.download_as_bytes()
    rotated = _rotate_png_bytes(orig_png, body.rotation)

    if body.overwrite_image:
        blob.upload_from_string(rotated, content_type="image/png")

    try:
        image_ref = _generate_signed_url_for_read(blob, minutes=10)
    except Exception:
        image_ref = _png_bytes_to_data_url(rotated)

    try:
        initial = await _extract_initial_for_page(image_url=image_ref)
    except Exception as e:
        raise HTTPException(500, f"LLM extract failed: {e}")

    if not isinstance(initial, dict):
        initial = {}
    for key, val in list(initial.items()):
        if isinstance(val, list):
            for r in val:
                if isinstance(r, dict):
                    r.setdefault("_page", local_page)

    try:
        filing_like = _initial_to_filing(initial, page_no=local_page)
        new_rows = _build_page_rows_for_write(
            filing_like, page_no=local_page, reset_validated=body.reset_validated
        )
    except Exception as e:
        raise HTTPException(500, f"Post-process to page rows failed: {e}")

    page_ref = filing_ref.collection("pages").document(str(local_page))
    batch = client.batch()
    for snap in page_ref.collection("rows").stream():
        batch.delete(snap.reference)
    for r in new_rows:
        batch.set(page_ref.collection("rows").document(), r)
    batch.set(
        page_ref,
        { "img": gs_url, "page_no": local_page, "rotation_applied": body.rotation, "updated": SERVER_TIMESTAMP },
        merge=True,
    )
    batch.commit()

    return ReextractResult(
        doc_id=doc_id, page=local_page, rows_written=len(new_rows), gs_url=gs_url, rotation_applied=body.rotation
    )

# ───────────────────── NEW fast preview endpoint ─────────────────────
class PreviewReq(BaseModel):
    slug: str
    global_page: int = Field(..., ge=1)
    rotation: int = 0
    overwrite_image: bool = True
    reset_validated: bool = True
    persist: bool = True  # enqueue background rotate+write

def _shape_rows_for_ui(filing_like: Dict[str, Any], *, page_no: int) -> List[Dict[str, Any]]:
    """
    Return rows shaped like /filings/{slug}/pages: {id,label,value,bbox,approved,row_type,columns,norm}
    Uses the same logic as filings_qc when building UI rows.
    """
    rows_ui: List[Dict[str, Any]] = []
    # Build writer rows, then convert to UI shape (cheap & deterministic)
    writer_rows = _build_page_rows_for_write(filing_like, page_no=page_no, reset_validated=True)
    for i, r in enumerate(writer_rows):
        rt = r.get("row_type")
        cols: Dict[str, Any] = {}
        if rt == "contribution":
            cols = {
                "date": r.get("date"),
                "donor": r.get("contributor") or r.get("name"),
                "address": r.get("address"),
                "city": r.get("city"), "state": r.get("state"), "zip": r.get("zip"),
                "type": r.get("type"), "amount": r.get("amount"),
            }
        elif rt == "expenditure":
            cols = {
                "date": r.get("date"),
                "payee": r.get("name") or r.get("payee"),
                "address": r.get("address"),
                "purpose": r.get("purpose"),
                "amount": r.get("amount"),
            }
        elif rt == "in_kind":
            cols = {
                "date": r.get("date"),
                "donor": r.get("contributor"),
                "description": r.get("description"),
                "value": r.get("value"),
            }
        elif rt == "debt":
            cols = {
                "date": r.get("date"),
                "creditor": r.get("creditor"),
                "purpose": r.get("purpose"),
                "balance": r.get("balance"),
            }

        rows_ui.append({
            "id"      : f"tmp_{i}",
            "label"   : cols.get("donor") or cols.get("payee") or r.get("row_type"),
            "value"   : r.get("amount") or r.get("value") or r.get("balance"),
            "bbox"    : r.get("bbox"),
            "approved": False,
            "row_type": rt,
            "columns" : {k: v for k, v in cols.items() if v not in ("", None, "")},
            "norm"    : None,  # optional; you can add the same norm block as filings_qc if you want
        })
    return rows_ui

@router.post("/reextract/preview")
async def reextract_preview(
    req: PreviewReq, background: BackgroundTasks, client: firestore.Client = Depends(get_firestore_client),
):
    t0 = time.perf_counter()
    logger.info("[reextract.preview] start slug=%s gpage=%d rotation=%d persist=%s",
                req.slug, req.global_page, req.rotation, req.persist)

    # resolve page
    t_resolve0 = time.perf_counter()
    doc_id, local_page = await _resolve_doc_and_local_page(
        client, doc_id=None, local_page=None, slug=req.slug, global_page=req.global_page
    )
    filing_ref = client.collection("filings").document(doc_id)
    filing_doc = filing_ref.get().to_dict() or {}
    files_map = filing_doc.get("files", {})
    gs_url = files_map.get(f"page_{local_page}")
    if not gs_url:
        raise HTTPException(404, f"Missing files.page_{local_page}")
    t_resolve1 = time.perf_counter()
    logger.info("[reextract.preview] resolved doc=%s local_page=%d resolve_ms=%.1f",
                doc_id, local_page, (t_resolve1 - t0) * 1000)

    # download and rotate IN MEMORY
    t_dl0 = time.perf_counter()
    storage = get_storage_client()
    bucket, blob_path = _parse_gs_url(gs_url)
    blob = storage.bucket(bucket).blob(blob_path)
    if not blob.exists():
        raise HTTPException(404, "Storage object not found for this page")
    orig_png = blob.download_as_bytes()
    t_dl1 = time.perf_counter()
    rotated  = _rotate_png_bytes(orig_png, req.rotation)
    t_rot1 = time.perf_counter()
    logger.info("[reextract.preview] storage_download_ms=%.1f bytes=%d rotate_ms=%.1f",
                (t_dl1 - t_dl0)*1000, len(orig_png), (t_rot1 - t_dl1)*1000)

    # LLM extract → filing-like → UI rows
    img_ref = _png_bytes_to_data_url(rotated)
    t_llm0 = time.perf_counter()
    try:
        initial     = await _extract_initial_for_page(image_url=img_ref)
        filing_like = _initial_to_filing(initial or {}, page_no=local_page)
        rows_ui     = _shape_rows_for_ui(filing_like, page_no=local_page)
    except Exception as e:
        logger.exception("[reextract.preview] extract_failed")
        raise HTTPException(500, f"Preview extract failed: {e}")
    t_llm1 = time.perf_counter()
    logger.info("[reextract.preview] llm_ms=%.1f shape_ms=%.1f",
                (t_llm1 - t_llm0)*1000, (time.perf_counter() - t_llm1)*1000)

    # Background persistence (rotated image + canonical writer rows)
    if req.persist:
        async def _bg():
            bg0 = time.perf_counter()
            try:
                up0 = time.perf_counter()
                if req.overwrite_image:
                    blob.upload_from_string(rotated, content_type="image/png")
                    dst_uri = gs_url
                else:
                    base, ext = (blob_path.rsplit(".", 1) + ["png"])[:2]
                    dst_name  = f"{base}_rot{req.rotation}.{ext}"
                    dst_blob  = storage.bucket(bucket).blob(dst_name)
                    dst_blob.upload_from_string(rotated, content_type="image/png")
                    dst_uri = f"gs://{bucket}/{dst_name}"
                    client.document(f"filings/{doc_id}").update({ f"files.page_{local_page}": dst_uri })
                up1 = time.perf_counter()

                writer_rows = _build_page_rows_for_write(
                    _initial_to_filing(initial or {}, page_no=local_page),
                    page_no=local_page, reset_validated=req.reset_validated
                )
                wr0 = time.perf_counter()
                page_ref = filing_ref.collection("pages").document(str(local_page))
                batch = client.batch()
                for snap in page_ref.collection("rows").stream():
                    batch.delete(snap.reference)
                for r in writer_rows:
                    batch.set(page_ref.collection("rows").document(), r)
                batch.set(page_ref, {
                    "img": dst_uri,
                    "page_no": local_page,
                    "rotation_applied": req.rotation,
                    "validated": False if req.reset_validated else False,
                    "updated": SERVER_TIMESTAMP,
                }, merge=True)
                batch.commit()
                wr1 = time.perf_counter()

                logger.info("[reextract.preview.bg] persisted doc=%s page=%d upload_ms=%.1f rows_ms=%.1f total_ms=%.1f",
                            doc_id, local_page, (up1-up0)*1000, (wr1-wr0)*1000, (wr1-bg0)*1000)
            except Exception as e:
                logger.exception("[reextract.preview.bg] persist_failed doc=%s page=%d", doc_id, local_page)

        background.add_task(_bg)
        logger.info("[reextract.preview] background_task_scheduled doc=%s page=%d", doc_id, local_page)
    else:
        logger.info("[reextract.preview] persist=false (no background writes)")

    logger.info("[reextract.preview] done preview_ms=%.1f", (time.perf_counter() - t0)*1000)

    return {
        "doc_id"     : doc_id,
        "page"       : req.global_page,
        "totalPages" : len([k for k in files_map.keys() if k.startswith("page_")]),
        "pageImg"    : files_map.get(f"page_{local_page}"),
        "rows"       : rows_ui,
        "files"      : files_map,
        "persisted"  : False,
        "timings_ms" : {
            "resolve": round((t_resolve1 - t0)*1000, 1),
            "download": round((t_dl1 - t_dl0)*1000, 1),
            "rotate": round((t_rot1 - t_dl1)*1000, 1),
            "llm": round((t_llm1 - t_llm0)*1000, 1),
            "preview_total": round((time.perf_counter() - t0)*1000, 1),
        }
    }


# Back-compat alias (keep the slow path alias if you still need it)
@router.post("/reextract", response_model=ReextractResult)
async def reextract_alias(
    body: ReextractRequest = Body(...),
    client: firestore.Client = Depends(get_firestore_client),
):
    return await rotate_and_reextract(body, client)
