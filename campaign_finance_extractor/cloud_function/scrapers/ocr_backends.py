# ── ocr_backends.py ───────────────────────────────────────────
from __future__ import annotations
from functools   import lru_cache
from typing      import Tuple
from PIL         import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

_MODEL_ID = "microsoft/trocr-small-handwritten"

@lru_cache(maxsize=1)           # one per *process*
def _load_trocr() -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    proc  = TrOCRProcessor.from_pretrained(_MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(_MODEL_ID).eval()
    return proc, model

def trocr_text(img: Image.Image) -> str:
    """Hand‑written OCR on a PIL crop → plain text."""
    proc, model = _load_trocr()          # cached after first call
    inputs  = proc(images=img.convert("RGB"), return_tensors="pt")
    ids     = model.generate(**inputs, max_length=64)
    return proc.batch_decode(ids, skip_special_tokens=True)[0]
