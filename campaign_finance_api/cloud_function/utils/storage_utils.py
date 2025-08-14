# utils/storage_utils.py  (validated against GCS docs)

from __future__ import annotations
import os, logging
from datetime import timedelta
from urllib.parse import quote_plus
from google.cloud import storage

logger = logging.getLogger(__name__)

# ───────── client helper ─────────
def get_storage_client() -> storage.Client:
    emu = os.getenv("STORAGE_EMULATOR_HOST")
    proj = os.getenv("GOOGLE_CLOUD_PROJECT", "dev-emulated-project")

    if emu:
        return storage.Client(
            project=proj,
            client_options={"api_endpoint": f"http://{emu}"},
            credentials=None,            # <- guarantees “no real creds”
        )
    return storage.Client(project=proj)   # prod path


# ───────── URI helper ────────────
def gs_to_http(gs_uri: str, *, signed: bool = True, expires: int = 900) -> str:
    """Return a browser‑fetchable URL for a gs:// object."""
    if not gs_uri.startswith("gs://"):
        return gs_uri

    bucket, blob = gs_uri[5:].split("/", 1)
    emu = os.getenv("STORAGE_EMULATOR_HOST")

    public_emu = os.getenv(                           # NEW –
        "PUBLIC_STORAGE_EMULATOR_HOST", "localhost:9199"
    )
    if emu:                                           # running against the emulator
        # use a host that the *browser* can resolve
        return f"http://{public_emu}/v0/b/{bucket}/o/{quote_plus(blob)}?alt=media"
    if not signed:  # public bucket
        return f"https://storage.googleapis.com/{bucket}/{quote_plus(blob)}"

    client = get_storage_client()
    return (client.bucket(bucket).blob(blob)
            .generate_signed_url(version="v4",
                                 method="GET",
                                 expiration=timedelta(seconds=expires)))
