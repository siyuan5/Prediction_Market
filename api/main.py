"""
FastAPI server for the prediction-market sim. Run from repo root:

  pip install fastapi uvicorn numpy pydantic
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

Endpoints fall into three groups:
  * Stateless: ``POST /api/simulate`` runs ``n_rounds`` and returns metrics + settlement.
  * Streaming: ``POST /api/simulate/stream`` emits one NDJSON line per round (UI live charts).
  * Session: ``/api/session/*`` keeps a ``SimulationEngine`` in memory for pause/step/shift/finish.

Optional: Ollama at http://127.0.0.1:11434 for LLM trader lines (``ollama pull <model>``).
Env: COMMENT_USE_LLM, OLLAMA_*, COMMENT_LLM_MAX, COMMENT_MAX_TOTAL (cap comments per run).
"""

from __future__ import annotations

import json
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, TypedDict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# src/ imports (same pattern as repo scripts)
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belief_init import BeliefSpec  # noqa: E402
from settlement import compute_settlement  # noqa: E402
from simulation_engine import SimulationEngine  # noqa: E402

from .llm_comments import generate_comment_text, llm_budget_initial

# Sparse synthetic chat: each (agent, round) gets a comment only with this probability.
COMMENT_PROB_PER_AGENT_ROUND = 0.01


def _comment_max_total() -> Optional[int]:
    """Max comments stored for one /simulate or one interactive session; None = no cap."""
    try:
        v = int(os.environ.get("COMMENT_MAX_TOTAL", "15").strip())
    except ValueError:
        v = 15
    if v <= 0:
        return None
    return v


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


def _make_engine(body: SimulateRequest) -> SimulationEngine:
    rho_values = body.rho_values if body.rho_values else [0.5, 1.0, 2.0]
    return SimulationEngine(
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


def _append_round_comments(
    engine: SimulationEngine,
    comment_rng: random.Random,
    comments: List[Dict[str, Any]],
    *,
    event_name: str,
    mechanism: str,
    llm_budget: Optional[List[int]],
) -> None:
    r = engine.round
    price = float(engine.get_state()["price"])
    cap = _comment_max_total()
    for row in engine.get_agents():
        if cap is not None and len(comments) >= cap:
            return
        if comment_rng.random() >= COMMENT_PROB_PER_AGENT_ROUND:
            continue
        text, source = generate_comment_text(
            event_name=event_name,
            mechanism=mechanism,
            belief=float(row["belief"]),
            agent_id=int(row["agent_id"]),
            round_num=int(r),
            market_yes_price=price,
            rng=comment_rng,
            llm_budget=llm_budget,
        )
        comments.append(
            {
                "round": r,
                "agent_id": row["agent_id"],
                "belief": float(row["belief"]),
                "yes_shares": float(row["shares"]),
                "text": text,
                "source": source,
            }
        )


class _SessionData(TypedDict):
    engine: SimulationEngine
    config: SimulateRequest
    comment_rng: random.Random
    comments: List[Dict[str, Any]]
    llm_budget: List[int]
    llm_budget_initial: int


# In-memory sessions (dev/demo); restart the server clears state.
_sessions: Dict[str, _SessionData] = {}


class SessionStepRequest(BaseModel):
    session_id: str
    rounds: int = Field(1, ge=1, le=500)


class SessionShiftRequest(BaseModel):
    session_id: str
    new_belief: Optional[float] = None
    delta: Optional[float] = None
    agent_ids: Optional[List[int]] = None
    rho_filter: Optional[float] = None

    @model_validator(mode="after")
    def exactly_one_belief_op(self) -> SessionShiftRequest:
        has_new = self.new_belief is not None
        has_delta = self.delta is not None
        if has_new == has_delta:
            raise ValueError("Provide exactly one of new_belief or delta")
        if has_new and not (0.01 <= float(self.new_belief) <= 0.99):
            raise ValueError("new_belief must be between 0.01 and 0.99")
        return self


class SessionIdBody(BaseModel):
    session_id: str


def _get_session(session_id: str) -> _SessionData:
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id")
    return s


def _session_snapshot(data: _SessionData, *, target_rounds: int, session_id: str) -> Dict[str, Any]:
    engine = data["engine"]
    return {
        "session_id": session_id,
        "target_rounds": target_rounds,
        "round": engine.round,
        "done": engine.round >= target_rounds,
        "state": engine.get_state(),
        "metrics": engine.get_metrics(),
        "comments": data["comments"],
        "agents": engine.get_agents(),
        "comment_sampling": {
            "probability_per_agent_per_round": COMMENT_PROB_PER_AGENT_ROUND,
            "max_comments_per_event": _comment_max_total(),
            "comments_so_far": len(data["comments"]),
            "llm_budget_initial": data["llm_budget_initial"],
            "llm_budget_remaining": data["llm_budget"][0],
        },
    }


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# --- Stateless full run: build engine, run all rounds, optional LLM comments, settle ---


@app.post("/api/simulate")
def simulate(body: SimulateRequest) -> Dict[str, Any]:
    engine = _make_engine(body)

    comment_rng = random.Random(body.seed + 17)
    comments: List[Dict[str, Any]] = []
    llm_cap = llm_budget_initial()
    llm_budget = [llm_cap]

    for _ in range(body.n_rounds):
        engine.run(1)
        _append_round_comments(
            engine,
            comment_rng,
            comments,
            event_name=body.event_name,
            mechanism=body.mechanism,
            llm_budget=llm_budget,
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
                "max_comments_per_event": _comment_max_total(),
                "comments_returned": len(comments),
                "llm_budget_initial": llm_cap,
                "llm_budget_remaining": llm_budget[0],
            },
            "config": body.model_dump(),
        }
    )


# --- NDJSON stream: same economics as /simulate, one tick payload per round for the UI ---


def _simulate_ndjson_chunks(body: SimulateRequest) -> Iterator[Dict[str, Any]]:
    """Yields tick dicts per round, then one done dict (same shape as /api/simulate plus type)."""
    engine = _make_engine(body)
    comment_rng = random.Random(body.seed + 17)
    comments: List[Dict[str, Any]] = []
    llm_cap = llm_budget_initial()
    llm_budget = [llm_cap]

    for _ in range(body.n_rounds):
        len_before = len(comments)
        engine.run(1)
        _append_round_comments(
            engine,
            comment_rng,
            comments,
            event_name=body.event_name,
            mechanism=body.mechanism,
            llm_budget=llm_budget,
        )
        new_comments = comments[len_before:]
        tick: Dict[str, Any] = {
            "type": "tick",
            "state": engine.get_state(),
            "mean_initial_belief": float(engine.mean_initial_belief),
            "append": {
                "price": float(engine.price_series[-1]),
                "mean_belief": float(engine.mean_belief_series[-1]),
                "error": float(engine.error_series[-1]),
                "trade_volume": float(engine.trade_volume[-1]),
            },
            "new_comments": new_comments,
            "comment_sampling": {
                "probability_per_agent_per_round": COMMENT_PROB_PER_AGENT_ROUND,
                "max_comments_per_event": _comment_max_total(),
                "comments_so_far": len(comments),
                "llm_budget_initial": llm_cap,
                "llm_budget_remaining": llm_budget[0],
            },
        }
        if engine.mechanism == "cda":
            tick["append_best_bid"] = engine.best_bid_series[-1] if engine.best_bid_series else None
            tick["append_best_ask"] = engine.best_ask_series[-1] if engine.best_ask_series else None
        yield tick

    agents_final = engine.get_agents()
    metrics = engine.get_metrics()
    state = engine.get_state()
    settlement = compute_settlement(
        agents_final,
        initial_cash=body.initial_cash,
        ground_truth=body.ground_truth,
        seed=body.seed,
    )
    yield {
        "type": "done",
        "event_name": body.event_name,
        "metrics": metrics,
        "agents_final": agents_final,
        "state": state,
        "settlement": settlement,
        "comments": comments,
        "comment_sampling": {
            "probability_per_agent_per_round": COMMENT_PROB_PER_AGENT_ROUND,
            "max_comments_per_event": _comment_max_total(),
            "comments_returned": len(comments),
            "llm_budget_initial": llm_cap,
            "llm_budget_remaining": llm_budget[0],
        },
        "config": body.model_dump(),
    }


@app.post("/api/simulate/stream")
def simulate_stream(body: SimulateRequest):
    """NDJSON stream: one {type:tick,...} per round, then {type:done,...} matching /api/simulate."""

    def ndjson_bytes() -> Iterator[bytes]:
        for chunk in _simulate_ndjson_chunks(body):
            line = json.dumps(_jsonable(chunk), separators=(",", ":")) + "\n"
            yield line.encode("utf-8")

    return StreamingResponse(
        ndjson_bytes(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --- Interactive session: step rounds, inject belief shocks, finish with settlement ---


@app.post("/api/session/start")
def session_start(body: SimulateRequest) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    engine = _make_engine(body)
    llm_cap = llm_budget_initial()
    data: _SessionData = {
        "engine": engine,
        "config": body,
        "comment_rng": random.Random(body.seed + 17),
        "comments": [],
        "llm_budget": [llm_cap],
        "llm_budget_initial": llm_cap,
    }
    _sessions[session_id] = data
    return _jsonable(_session_snapshot(data, target_rounds=body.n_rounds, session_id=session_id))


@app.post("/api/session/step")
def session_step(body: SessionStepRequest) -> Dict[str, Any]:
    data = _get_session(body.session_id)
    engine = data["engine"]
    cfg = data["config"]
    remaining = max(0, cfg.n_rounds - engine.round)
    if remaining == 0:
        snap = _session_snapshot(data, target_rounds=cfg.n_rounds, session_id=body.session_id)
        snap["rounds_advanced"] = 0
        return _jsonable(snap)

    to_run = min(body.rounds, remaining)
    for _ in range(to_run):
        engine.run(1)
        _append_round_comments(
            engine,
            data["comment_rng"],
            data["comments"],
            event_name=cfg.event_name,
            mechanism=cfg.mechanism,
            llm_budget=data["llm_budget"],
        )

    snap = _session_snapshot(data, target_rounds=cfg.n_rounds, session_id=body.session_id)
    snap["rounds_advanced"] = to_run
    return _jsonable(snap)


@app.post("/api/session/shift")
def session_shift(body: SessionShiftRequest) -> Dict[str, Any]:
    data = _get_session(body.session_id)
    engine = data["engine"]
    cfg = data["config"]
    kw: Dict[str, Any] = {}
    if body.new_belief is not None:
        kw["new_belief"] = body.new_belief
    else:
        kw["delta"] = body.delta
    if body.agent_ids is not None:
        kw["agent_ids"] = body.agent_ids
    if body.rho_filter is not None:
        kw["rho_filter"] = body.rho_filter
    event = engine.shift_beliefs(**kw)
    snap = _session_snapshot(data, target_rounds=cfg.n_rounds, session_id=body.session_id)
    snap["shift_event"] = event
    return _jsonable(snap)


@app.post("/api/session/finish")
def session_finish(body: SessionIdBody) -> Dict[str, Any]:
    data = _get_session(body.session_id)
    cfg = data["config"]
    engine = data["engine"]
    agents_final = engine.get_agents()
    state = engine.get_state()
    metrics = engine.get_metrics()
    settlement = compute_settlement(
        agents_final,
        initial_cash=cfg.initial_cash,
        ground_truth=cfg.ground_truth,
        seed=cfg.seed,
    )
    del _sessions[body.session_id]
    return _jsonable(
        {
            "event_name": cfg.event_name,
            "metrics": metrics,
            "agents_final": agents_final,
            "state": state,
            "settlement": settlement,
            "comments": data["comments"],
            "comment_sampling": {
                "probability_per_agent_per_round": COMMENT_PROB_PER_AGENT_ROUND,
                "max_comments_per_event": _comment_max_total(),
                "comments_returned": len(data["comments"]),
                "llm_budget_initial": data["llm_budget_initial"],
                "llm_budget_remaining": data["llm_budget"][0],
            },
            "config": cfg.model_dump(),
        }
    )


@app.delete("/api/session/{session_id}")
def session_delete(session_id: str) -> Dict[str, str]:
    _sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
