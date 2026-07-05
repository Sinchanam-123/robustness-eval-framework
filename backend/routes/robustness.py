"""GET /api/robustness-index — one overall score per model + per-perturbation breakdown.

Reads via evaluation.robustness_index — not recomputed here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter

import config
from backend.schemas import RobustnessIndexEntry, RobustnessIndexResponse
from evaluation.robustness_index import compute_robustness_index

router = APIRouter()


@router.get("/api/robustness-index", response_model=RobustnessIndexResponse)
def get_robustness_index() -> RobustnessIndexResponse:
    overall_df, per_perturbation_df, _ = compute_robustness_index()

    entries = []
    for _, row in overall_df.iterrows():
        breakdown = (
            per_perturbation_df[per_perturbation_df["model"] == row["model"]]
            .set_index("perturbation")["robustness_index"]
            .to_dict()
        )
        entries.append(
            RobustnessIndexEntry(
                model=row["model"],
                overall_robustness_index=row["overall_robustness_index"],
                per_perturbation=breakdown,
            )
        )

    return RobustnessIndexResponse(metric=config.ROBUSTNESS_METRIC, robustness_index=entries)
