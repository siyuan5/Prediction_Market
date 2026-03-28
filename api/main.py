"""
FastAPI server for the prediction-market sim. Run from repo root:

  pip install fastapi uvicorn numpy pydantic
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# src/ imports (same pattern as repo scripts)
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belief_init import BeliefSpec  # noqa: E402
from settlement import compute_settlement  # noqa: E402
from simulation_engine import SimulationEngine  # noqa: E402

from .comments import pick_filler_comment

# Low chance each agent speaks in a given round; many agents × many rounds → plenty of comments overall.
COMMENT_PROB_PER_AGENT_ROUND = 0.01


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (float, int, str, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


class SimulateRequest(BaseModel):
    mechanism: Literal["lmsr", "cda"] = "lmsr"
    event_name: str = Field("Untitled event", max_length=200)
    seed: int = 42
    ground_truth: float = Field(0.70, ge=0.01, le=0.99)
    n_agents: int = Field(50, ge=2, le=500)
    n_rounds: int = Field(100, ge=1, le=2000)
    initial_cash: float = Field(100.0, gt=0)
    b: float = Field(100.0, gt=0)
    initial_price: float = Field(0.5, ge=0.01, le=0.99)
    belief_mode: Literal["gaussian", "uniform", "fixed", "bimodal"] = "gaussian"
    belief_sigma: float = Field(0.10, ge=0.0, le=0.5)
    belief_fixed: float = Field(0.70, ge=0.01, le=0.99)
    rho_values: Optional[List[float]] = None

    @field_validator("event_name", mode="before")
    @classmethod
    def _normalize_event_name(cls, v: object) -> str:
        if v is None:
            return "Untitled event"
        s = str(v).strip()
        return s if s else "Untitled event"


app = FastAPI(title="Prediction Market Sim API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _belief_spec(body: SimulateRequest) -> BeliefSpec:
    if body.belief_mode == "gaussian":
        return BeliefSpec(mode="gaussian", sigma=body.belief_sigma)
    if body.belief_mode == "uniform":
        return BeliefSpec(mode="uniform")
    if body.belief_mode == "fixed":
        return BeliefSpec(mode="fixed", fixed_value=body.belief_fixed)
    return BeliefSpec(mode="bimodal")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/simulate")
def simulate(body: SimulateRequest) -> Dict[str, Any]:
    rho_values = body.rho_values if body.rho_values else [0.5, 1.0, 2.0]
    engine = SimulationEngine(
        mechanism=body.mechanism,
        phase=2,
        seed=body.seed,
        ground_truth=body.ground_truth,
        n_agents=body.n_agents,
        initial_cash=body.initial_cash,
        rho_values=rho_values,
        belief_spec=_belief_spec(body),
        b=body.b,
        initial_price=body.initial_price,
    )

    comment_rng = random.Random(body.seed + 17)
    comments: List[Dict[str, Any]] = []

    for _ in range(body.n_rounds):
        engine.run(1)
        r = engine.round
        for row in engine.get_agents():
            if comment_rng.random() >= COMMENT_PROB_PER_AGENT_ROUND:
                continue
            comments.append(
                {
                    "round": r,
                    "agent_id": row["agent_id"],
                    "belief": float(row["belief"]),
                    "yes_shares": float(row["shares"]),
                    "text": pick_filler_comment(
                        float(row["belief"]),
                        int(row["agent_id"]),
                        int(r),
                        comment_rng,
                    ),
                }
            )

    metrics = engine.get_metrics()
    agents_final = engine.get_agents()
    state = engine.get_state()
    settlement = compute_settlement(
        agents_final,
        initial_cash=body.initial_cash,
        ground_truth=body.ground_truth,
        seed=body.seed,
    )

    return _jsonable(
        {
            "event_name": body.event_name,
            "metrics": metrics,
            "agents_final": agents_final,
            "state": state,
            "settlement": settlement,
            "comments": comments,
            "comment_sampling": {
                "probability_per_agent_per_round": COMMENT_PROB_PER_AGENT_ROUND,
            },
            "config": body.model_dump(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
