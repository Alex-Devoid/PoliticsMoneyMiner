# cloud_function/routes/finance_states.py  (NEW small file)
from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from google.cloud import firestore                     # type: ignore
from cloud_function.utils.firestore_utils import get_firestore_client

router = APIRouter(prefix="/finance")   # <── prefix puts the route at /finance/…

@router.get("/states", response_model=List[str])
async def list_finance_states(client = Depends(get_firestore_client)):
    """
    Return distinct 2-letter state / jurisdiction codes that have finance data.
    """
    try:
        docs = (client.collection("finance_summary")
                       .select(["jurisdiction"]).stream())

        states = { (d.to_dict().get("jurisdiction") or "").upper()
                   for d in docs }

        return sorted(s for s in states if s)
    except Exception as exc:                               # noqa: BLE001
        raise HTTPException(500, f"Failed to fetch states: {exc}") from exc
