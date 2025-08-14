# cloud_function/utils/storage_utils.py
"""
Utility for initialising a Google Cloud **Storage** client that automatically
connects either to the Firebase **Storage emulator** (when
STORAGE_EMULATOR_HOST is set) or to real GCS in production, matching the
pattern used in firestore_utils.py.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from google.auth.exceptions import DefaultCredentialsError

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG)
_LOG = logging.getLogger(__name__)


def get_storage_client() -> storage.Client:
    """
    Return a configured Cloud-Storage client.

    • If STORAGE_EMULATOR_HOST is defined → connect to the emulator with
      AnonymousCredentials().
    • Otherwise use Application Default Credentials (ADC) for production.
      In CI / local runs *without* ADC we fall back to AnonymousCredentials so
      code continues to run (read-only or will error on writes).
    """

    emulator_host = os.getenv("STORAGE_EMULATOR_HOST", "")
    project_id    = os.getenv("GOOGLE_CLOUD_PROJECT", "dev-emulated-project")

    _LOG.debug(
        "Environment: STORAGE_EMULATOR_HOST=%s · GOOGLE_CLOUD_PROJECT=%s",
        emulator_host or "∅",
        project_id,
    )

    if emulator_host:
        _LOG.info(
            "Connecting to Storage EMULATOR at %s (project=%s)",
            emulator_host,
            project_id,
        )
        client = storage.Client(
            project       = project_id,
            
            client_options= {"api_endpoint": emulator_host},
        )
    else:
        _LOG.info("Connecting to real Cloud Storage (project=%s)", project_id)
        try:
            client = storage.Client(project=project_id)     # uses ADC
        except DefaultCredentialsError:
            _LOG.warning(
                "⚠️  No Application Default Credentials found – "
                "falling back to AnonymousCredentials (may be RO only)."
            )
            client = storage.Client(
                project     = project_id,
                credentials = AnonymousCredentials(),
            )

    _LOG.debug("Storage client initialised; project=%s", client.project)
    return client
