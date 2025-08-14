"""
cloud_function/main.py
FastAPI entry-point for Democracy Dollars backend
──────────────────────────────────────────────────
* Creates the app
* Configures logging & global CORS
* Exposes health-check
* Mounts feature routers (newsletter, sheets, Sedgwick scraper, …)
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── project-internal helpers ──────────────────────────────────────────
from cloud_function.database import get_engine, get_session
from cloud_function.config import settings  # Pydantic BaseSettings object
# from cloud_function.services_frontend import router as frontend_router
# from cloud_function.services_sheet import router as sheet_router
from cloud_function.routes import sedgwick as sedgwick_router
from cloud_function.routes import kentucky as ky_router
# from cloud_function.process_data import router as newsletter_router

# ── global logging ----------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
logger.info("Starting backend with log-level %s", LOG_LEVEL)

# ── create FastAPI instance -------------------------------------------
app = FastAPI(
    title="Democracy Dollars API",
    version="1.0.0",
    description=(
        "Backend micro-service for the journalism data pipeline & front-end.\n\n"
        "Includes newsletter processing, Google Sheets utilities, and the "
        "Sedgwick County campaign-finance scraper."
    ),
    contact={"name": "Engineering", "email": "eng@example.org"},
    docs_url="/",
)

# ── CORS (adjust origins in .env / config.py) -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,     # list[str]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB startup/teardown hooks (optional) ------------------------------
@app.on_event("startup")
async def _startup() -> None:
    engine = get_engine()
    logger.info("DB engine created: %s", engine)
    # if you need alembic migrations on boot, trigger them here

@app.on_event("shutdown")
async def _shutdown() -> None:
    get_engine().dispose()
    logger.info("DB engine disposed")

# ── health-check ------------------------------------------------------
@app.get("/healthz", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}

# ── router mount-point ------------------------------------------------
# app.include_router(frontend_router)          # /frontend/…
# app.include_router(sheet_router)             # /sheets/…
# app.include_router(newsletter_router)        # /process-newsletter
app.include_router(sedgwick_router.router)   # /sedgwick/scrape
app.include_router(ky_router.router)
# ── global exception handler (optional) -------------------------------
@app.exception_handler(Exception)
async def unhandled_exceptions(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
