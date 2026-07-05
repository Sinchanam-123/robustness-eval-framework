"""GET /api/results — aggregated mean +- std per (model, perturbation, severity).

Reads via evaluation.aggregate — the same aggregation Phase 4/5 use, not
recomputed here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter

import config
from backend.schemas import ResultRow, ResultsResponse
from evaluation.aggregate import aggregate_results

router = APIRouter()


@router.get("/api/results", response_model=ResultsResponse)
def get_results() -> ResultsResponse:
    aggregated_df = aggregate_results()
    rows = [ResultRow(**row) for row in aggregated_df.to_dict(orient="records")]
    return ResultsResponse(tested_severities=config.SEVERITIES, results=rows)
