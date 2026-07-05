"""GET /api/perturbations — perturbation types + scope text.

Descriptions are presentation text, but every parameter they quote (impute
strategy, outlier scale, shift feature) is read from config.py, not
hardcoded here — if those constants change, this text stays accurate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter

import config
from backend.schemas import PerturbationInfo, PerturbationsResponse

router = APIRouter()

_DESCRIPTIONS = {
    "noise": (
        "Adds zero-mean Gaussian noise to each numeric feature, scaled to "
        "that feature's own standard deviation times severity."
    ),
    "missing_values": (
        f"Randomly nulls numeric values at rate = severity, then imputes "
        f"using the '{config.MISSING_VALUE_IMPUTE_STRATEGY}' strategy."
    ),
    "outliers": (
        f"Multiplies the numeric values of a severity-fraction of rows by "
        f"{config.OUTLIER_SCALE}x."
    ),
    "covariate_shift": (
        f"Shifts the '{config.SHIFT_FEATURE}' feature's mean by "
        f"severity * std. Covariate shift only — this does not simulate "
        f"label shift or concept shift."
    ),
}


@router.get("/api/perturbations", response_model=PerturbationsResponse)
def get_perturbations() -> PerturbationsResponse:
    perturbations = [
        PerturbationInfo(name=name, description=_DESCRIPTIONS[name])
        for name in config.PERTURBATION_NAMES
    ]
    return PerturbationsResponse(perturbations=perturbations)
