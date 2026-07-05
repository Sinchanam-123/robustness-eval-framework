"""GET /api/models — model list + baseline (severity=0) metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from fastapi import APIRouter

import config
from backend.schemas import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/api/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    baseline_df = pd.read_csv(config.BASELINE_METRICS_PATH)
    models = [ModelInfo(**row) for row in baseline_df.to_dict(orient="records")]
    return ModelsResponse(models=models)
