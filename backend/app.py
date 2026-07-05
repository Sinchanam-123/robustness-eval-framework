"""
Phase 6, Day 11 — FastAPI backend entrypoint.

Presentation layer only: every route reads results.csv / baseline_metrics.csv
(directly, or via the evaluation/insights modules already built on top of
them). No ML logic — no model loading, no predicting, no metric computation
— is reimplemented here.

Run: uvicorn backend.app:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import insights, models, perturbations, results, robustness

app = FastAPI(title="Robustness Evaluation Framework API")

# Local React dev server only (Vite default 5173, CRA default 3000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(perturbations.router)
app.include_router(results.router)
app.include_router(robustness.router)
app.include_router(insights.router)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}
