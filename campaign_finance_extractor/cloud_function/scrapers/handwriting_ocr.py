# ─── handwriting_ocr.py (new helper module) ────────────────────────────────
from pathlib import Path
from PIL import Image
import torch, json

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
#     ↳ change to `DonutProcessor` / `AutoModelForVision2Seq`
#       if you prefer NAVER-Donut instead of TrOCR.

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face model hub refs
# _MODEL_ID = "microsoft/trocr-base-handwritten"          # handwriting-tuned
_MODEL_ID = "microsoft/trocr-small-handwritten" 
# _MODEL_ID = "naver-clova-ix/donut-base-finetuned-zh-handwriting"  # example

_processor = TrOCRProcessor.from_pretrained(_MODEL_ID)
_model     = VisionEncoderDecoderModel.from_pretrained(_MODEL_ID).to(_DEVICE)
_model.eval()


@torch.no_grad()
def ocr_handwritten(img: Image.Image) -> list[dict]:
    """
    Return a *token-level* list identical to `_extract_ocr_spans`
    so it can be merged seamlessly with your existing tokens.
    """
    pixel_values = _processor(images=img, return_tensors="pt").pixel_values.to(_DEVICE)
    generated_ids = _model.generate(pixel_values, max_length=256)
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # naive token->bbox: we only get line-level text from TrOCR.
    # You can refine this later with a lightweight layout-parser.
    w, h = img.size
    return [{
        "page": 1,        # caller should overwrite with real page number
        "text": t,
        "x0": 0, "y0": 0, "x1": w, "y1": h      # full-page bbox fallback
    } for t in text.split()]
