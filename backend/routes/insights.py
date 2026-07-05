"""GET /api/insights — generated insight strings.

Reads via insights.generator — not recomputed here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter

from backend.schemas import InsightsResponse
from insights.generator import generate_insights

router = APIRouter()


@router.get("/api/insights", response_model=InsightsResponse)
def get_insights() -> InsightsResponse:
    return InsightsResponse(insights=generate_insights())
