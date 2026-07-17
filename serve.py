"""Optional bonus: serve the pipeline as a POST /extract JSON API.

    CONFIG=configs/baseline.yaml uvicorn serve:app --host 0.0.0.0 --port 8000
    curl -s localhost:8000/extract -H 'content-type: application/json' \
        -d '{"text": "Bệnh nhân sốt cao, không ho. Amlodipine 10 mg."}'

The same build_pipeline() core that run.py uses; handy for a live demo.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from medextract.config import load_config
from medextract.pipeline import build_pipeline

app = FastAPI(title="medextract")
_pipeline = None


class Req(BaseModel):
    text: str


def _get():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline(load_config(os.environ.get("CONFIG", "configs/baseline.yaml")))
    return _pipeline


@app.post("/extract")
def extract(req: Req):
    return _get().run_text(req.text)
