# cloud_function/main.py  –  drop-in replacement
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

# ─── application routers ─────────────────────────────────────────
from cloud_function.routes.finance_query import router as finance_query_router
from cloud_function.routes import finance_states  # <-- keep as module; we need .router
from cloud_function.routes.filings_qc import router as qc_router 
from cloud_function.routes.reextract_page import router as reextract_router
# ─── std / 3p libs used elsewhere in this file (unchanged) ──────
import jwt
import requests  # noqa: F401  (left for parity with original)

# ─── logging config (unchanged) ─────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ─── FastAPI app & CORS (unchanged) ─────────────────────────────
app = FastAPI()

ALLOWED_ORIGINS = [
    "https://argus.mcclatchy.com",                              # prod
    "https://argus-frontend-963969693847.us-central1.run.app",  # Cloud Run
    "http://localhost:3000",                                    # local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; tighten in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─── simple OPTIONS handler for CORS pre-flight ─────────────────
preflight_router = APIRouter()

@preflight_router.options("/{full_path:path}")
async def preflight_handler() -> JSONResponse:
    return JSONResponse("Preflight OK")

app.include_router(preflight_router)

# ─── env vars & helpers (unchanged) ─────────────────────────────
load_dotenv()

def log_request_info(req: Request) -> None:  # noqa: D401
    important = {
        k: v
        for k, v in req.headers.items()
        if k.lower()
        in {
            "authorization",
            "x-goog-iap-jwt-assertion",
            "x-goog-authenticated-user-email",
            "x-goog-authenticated-user-id",
            "origin",
            "referer",
        }
    }
    logger.info("Incoming %s %s • IP=%s • hdr=%s",
                req.method,
                req.url,
                req.client.host if req.client else "unknown",
                important)

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    logger.info("Incoming request headers: %s", dict(request.headers))
    return await call_next(request)

# ─── trivial root route (unchanged) ─────────────────────────────
@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI app!"}

# ─── include feature routers ------------------------------------
# 1) campaign-finance query endpoints
app.include_router(finance_query_router, prefix="/finance")

# 2) campaign-finance STATES endpoint
#    finance_states.router itself already has prefix="/finance".
#    If you remove that prefix inside finance_states.py, add it here instead.
app.include_router(finance_states.router)
app.include_router(qc_router)
app.include_router(reextract_router)
# ─── helper to print routes on startup (unchanged) ──────────────
def print_routes(fapp: FastAPI) -> None:
    print("\nRegistered Endpoints:")
    for r in fapp.routes:
        if isinstance(r, APIRoute):
            print(f"{r.methods} {r.path}")

print_routes(app)

# ─── global exception handler (unchanged) ───────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    logger.error("Full traceback:\n%s", traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"An unexpected error occurred: {exc}",
            "request_method": request.method,
            "request_url": str(request.url),
        },
    )
