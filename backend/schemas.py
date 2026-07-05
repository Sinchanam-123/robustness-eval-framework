"""
Pydantic response models for the read-only API.

The web app is a presentation layer only: every field here mirrors a column
already produced by the offline pipeline (results.csv, baseline_metrics.csv,
or the evaluation/insights modules built on top of them) — nothing is
computed inside these schemas or the route handlers that use them.
"""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class PerturbationInfo(BaseModel):
    name: str
    description: str


class PerturbationsResponse(BaseModel):
    perturbations: list[PerturbationInfo]


class ResultRow(BaseModel):
    model: str
    perturbation: str
    severity: float
    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float


class ResultsResponse(BaseModel):
    tested_severities: list[float]
    results: list[ResultRow]


class RobustnessIndexEntry(BaseModel):
    model: str
    overall_robustness_index: float
    per_perturbation: dict[str, float]


class RobustnessIndexResponse(BaseModel):
    metric: str
    robustness_index: list[RobustnessIndexEntry]


class InsightsResponse(BaseModel):
    insights: list[str]
