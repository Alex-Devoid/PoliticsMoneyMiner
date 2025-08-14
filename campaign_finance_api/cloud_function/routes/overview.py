"""
cloud_function/routes/overview.py
────────────────────────────────────────────────────────────────────────
Returns a high-level snapshot:

    {
      "states_count": 16,
      "boards_count": 214,
      "states"      : ["ca", "fl", "il", …]          // lowercase, sorted
    }

The query projects only the two fields we care about and keeps
distinct values in memory, so it is far lighter than pulling
whole documents.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Set
from google.cloud import firestore

from cloud_function.utils.firestore_utils import get_firestore_client

router = APIRouter()


class OverviewOut(BaseModel):
    states_count: int
    boards_count: int
    states: List[str]


@router.get("/overview", response_model=OverviewOut)
async def get_overview(client=Depends(get_firestore_client)):
    try:
        # projection → only 'state' and 'committee_name' come over the wire
        qs = (
            client.collection("scraped_meetings")
                  .select(["state", "committee_name"])
        )

        states: Set[str]     = set()
        committees: Set[str] = set()

        for doc in qs.stream():
            data = doc.to_dict()
            if s := data.get("state"):
                states.add(s.lower())
            if c := data.get("committee_name"):
                committees.add(c)

        return OverviewOut(
            states_count=len(states),
            boards_count=len(committees),
            states=sorted(states)
        )

    except Exception as e:           # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))
