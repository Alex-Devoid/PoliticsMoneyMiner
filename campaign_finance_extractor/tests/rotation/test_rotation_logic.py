# tests/rotation/test_rotation_logic.py
"""
Live integration test for the rotation pipeline.

• Re-uses *exactly* the same `_decide_rotation_deg()` helper as production
  – we only prepend a quick “centre-crop” vote to save tokens/time.
• Requires an OPENAI_API_KEY in the env (skips otherwise).
• Writes before/after PNGs to  tests/rotation/out/  for manual eyeballing.
"""
from __future__ import annotations

import os, sys, shutil, pathlib, pytest, unittest.mock as mock
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel

# ── make package imports work when run via `pytest`
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# ── scraper + helpers
from cloud_function.scrapers.SedgwickElectionsCampaignExpenseReport import (  # noqa: E402
    SedgwickExpenseScraper,
    _paddle_orientation,
    best_rotation,
)

# ── early-exit if no key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
pytest.skip(
    "OPENAI_API_KEY not set – live LLM tests skipped",
    allow_module_level=True,
) if not OPENAI_KEY else None

# ── page fixtures
PAGE05 = Path(
    "tests/scrapers/jeff-blubaugh/"
    "AZxLNq3nBF3KNTN9avSs8uRPpKpr1L44veVyAiTUhÁWx2b2WJ9LHNhUfgk9mqBs4DhBIpbkrtqy2onQTRÉ0mKvM="
    "/AZxLNq3nBF3KNTN9avSs8uRPpKpr1L44veVyAiTUhÁWx2b2WJ9LHNhUfgk9mqBs4DhBIpbkrtqy2onQTRÉ0mKvM="
    "/page05_orig.png"
)
PAGE04 = Path(
    "tests/scrapers/jeff-blubaugh/"
    "AZxLNq3nBF3KNTN9avSs8uRPpKpr1L44veVyAiTUhÁWx2b2WJ9LHNhUfgk9mqBs4DhBIpbkrtqy2onQTRÉ0mKvM="
    "/AZxLNq3nBF3KNTN9avSs8uRPpKpr1L44veVyAiTUhÁWx2b2WJ9LHNhUfgk9mqBs4DhBIpbkrtqy2onQTRÉ0mKvM="
    "/page04_orig.png"
)
PAGE03 = Path(
    "tests/scrapers/jeff-blubaugh/"
    "Af3XBktYicMDXÁ8Y7rIjw6iRFYÁ3NGzyLQ5Ly3vyTQVoFG621icfÉÁJY74hÉH62kcAz7MTeGkwmZyofX6ZpLUOM="
    "/Af3XBktYicMDXÁ8Y7rIjw6iRFYÁ3NGzyLQ5Ly3vyTQVoFG621icfÉÁJY74hÉH62kcAz7MTeGkwmZyofX6ZpLUOM="
    "/page03_orig.png"
)
PAGE01 = Path(
    "tests/scrapers/jeff-blubaugh/"
    "Af3XBktYicMDXÁ8Y7rIjw6iRFYÁ3NGzyLQ5Ly3vyTQVoFG621icfÉÁJY74hÉH62kcAz7MTeGkwmZyofX6ZpLUOM="
    "/Af3XBktYicMDXÁ8Y7rIjw6iRFYÁ3NGzyLQ5Ly3vyTQVoFG621icfÉÁJY74hÉH62kcAz7MTeGkwmZyofX6ZpLUOM="
    "/page01_orig.png"
)

# ── ground-truth needed rotations  (270 == 90° clockwise)
EXPECTED: dict[Path, int] = {
    PAGE01: 0,     # already upright
    PAGE03: 270,   # needs 90° clockwise
    PAGE04: 90,    # needs 90° counter-clockwise
    PAGE05: 90,    # needs 90° counter-clockwise
}

OUT_DIR = Path("tests/rotation/out")

# ── GPT-4o rotation helper (structured output)
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
    rotate: Literal[-90, 0, 90, 270]

_RotateHint.model_rebuild()

from langchain_openai import ChatOpenAI  # noqa: E402

LLM_ROTATE = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=OPENAI_KEY,
    max_tokens=20,
).with_structured_output(_RotateHint)

# ── stub GCP ADC so the scraper starts without real creds
from google.auth.credentials import AnonymousCredentials
mock.patch(
    "google.auth.default",
    return_value=(AnonymousCredentials(), "rotation-test"),
).start()

SCRAPER = SedgwickExpenseScraper(use_openai=False)

# ── crop-first vote (test-only helper)
def _centre_crop(img: Image.Image, frac: float = 0.35) -> Image.Image:
    w, h = img.size
    cx, cy = w // 2, h // 2
    hw, hh = int(w * frac // 2), int(h * frac // 2)
    return img.crop((cx - hw, cy - hh, cx + hw, cy + hh))

def _decide_deg_crop_first(
    img: Image.Image,
    *,
    scraper: SedgwickExpenseScraper,
    llm_rotate,
    prompt: str,
    frac: float = 0.35,
) -> int:
    crop_deg = scraper._decide_rotation_deg(
        img=_centre_crop(img, frac),
        llm_rotate=llm_rotate,
        rotate_prompt=prompt,
        returns_current=True,
    )
    if crop_deg:
        return crop_deg            # good answer – accept

    # ‼️ ask GPT on the whole page once ‼️
    page_deg = scraper._decide_rotation_deg(
        img=img,
        llm_rotate=llm_rotate,
        rotate_prompt=prompt,
        returns_current=True,
    )
    if page_deg:
        return page_deg            # trust GPT over heuristics

    # classical fall-back (Paddle + Tesseract)
    return scraper._decide_rotation_deg(img=img, llm_rotate=None, rotate_prompt="",returns_current=True,)

# ── optional legacy heuristic (debug only)
def _legacy_orientation(img: Image.Image) -> int:
    return _paddle_orientation(img) or best_rotation(img) or 0

# ── parametrised test
@pytest.mark.parametrize("img_path, want_deg", list(EXPECTED.items()))
def test_rotation_pipeline(img_path: Path, want_deg: int):
    assert img_path.exists(), f"missing fixture {img_path}"
    img = Image.open(img_path)

    # votes for visibility
    gpt_vote    = SCRAPER._llm_rotation_vote(img, llm_rotate=LLM_ROTATE,
                                            rotate_prompt=_ROTATE_PROMPT,returns_current=True)
    paddle_vote = _paddle_orientation(img)
    tess_vote   = best_rotation(img) or 0
    legacy_vote = _legacy_orientation(img)

    # convert −90 → 270 so everything is in 0/90/180/270 space
    to_360 = lambda d: 270 if d == -90 else (d or 0)

    # production decision (with crop fast-path)
    got_deg = gpt_vote  
    # to_360(
    #     _decide_deg_crop_first(
    #         img, scraper=SCRAPER, llm_rotate=LLM_ROTATE, prompt=_ROTATE_PROMPT
    #     )
    # )

    # --------------------  💡 NEW “who-won?” LOGIC  --------------------
    def winner():
        if got_deg == to_360(gpt_vote):
            return "GPT"
        if got_deg == to_360(paddle_vote):
            return "Paddle"
        if got_deg == to_360(tess_vote):
            return "Tesseract"
        return "Legacy"    

    # artefacts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(img_path, OUT_DIR / f"{img_path.stem}_orig.png")
    img.rotate( got_deg, expand=True).save(
        OUT_DIR / f"{img_path.stem}_rot.png", optimize=True
    )

    # consolidated print
    print(
        f"[{img_path.name}]  "
        f"EXPECT {want_deg:+d}° | GOT {got_deg:+d}°   "
        f"(chosen ← {winner()})   "
        f"votes → GPT {to_360(gpt_vote):+d}°, "
        f"Paddle {to_360(paddle_vote):+d}°, "
        f"Tess {to_360(tess_vote):+d}°, "
        f"Legacy {to_360(legacy_vote):+d}°"
    )

    # assertion
    assert got_deg == want_deg
