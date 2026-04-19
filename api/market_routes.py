"""
HTTP API for the persistent SQLite market (LMSR/CDA) and autonomous agent runners.

Mounted under ``/api/market/*``. Uses :class:`MarketService` for trades and beliefs;
background threads run :class:`autonomous_agent.AutonomousAgent` clients that poll
the same API (loopback) when ``POST .../start`` is called.

Environment:
  MARKET_DB_PATH — SQLite file for market state (default: ``data/markets.sqlite``).
  AUTONOMOUS_API_BASE — Base URL autonomous threads use (default: ``http://127.0.0.1:8000/api``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

# ``app/`` (market_service) and ``src/`` (beliefs, agents)
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "app", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent_runner import AgentRunner  # noqa: E402
from market_service import MarketService  # noqa: E402
from personality import DEFAULT_POPULATION_DIST, sample_personality  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])
agents_router = APIRouter(tags=["agents"])
_DEFAULT_SIGNAL_SENSITIVITY = 0.50
_DEFAULT_STUBBURNNESS = 0.30

# --- Singleton service (reset in tests via ``reset_market_runtime``) ---
_market_service: Optional[MarketService] = None
_agent_runner: Optional[AgentRunner] = None
# market_id -> metadata from create()
_market_initial_cash: Dict[int, float] = {}


def reset_market_runtime() -> None:
    """Clear singleton and runners (used by tests only)."""
    global _market_service, _agent_runner
    if _agent_runner is not None:
        _agent_runner.shutdown()
        _agent_runner = None
    if _market_service is not None:
        _market_service.close()
        _market_service = None
    _market_initial_cash.clear()


def _db_path() -> str:
    default = _ROOT / "data" / "markets.sqlite"
    p = os.environ.get("MARKET_DB_PATH", str(default))
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return p


def get_market_service() -> MarketService:
    global _market_service
    if _market_service is None:
        _market_service = MarketService(_db_path())
    return _market_service


def _autonomous_api_base() -> str:
    return os.environ.get("AUTONOMOUS_API_BASE", "http://127.0.0.1:8000/api").rstrip("/")


def get_agent_runner() -> AgentRunner:
    global _agent_runner
    if _agent_runner is None:
        _agent_runner = AgentRunner(
            api_base_url=_autonomous_api_base(),
            market_service=get_market_service(),
        )
    return _agent_runner


# --- Pydantic request bodies (match FINAL_PHASE_TASKS contracts) ---


class MarketCreateRequest(BaseModel):
    mechanism: str = "lmsr"
    ground_truth: float = Field(0.70, ge=0.01, le=0.99)
    b: float = Field(100.0, gt=0)
    title: str = "Autonomous market"
    description: str = ""
    tick_size: float = 0.0001
    min_price: float = 0.001
    max_price: float = 0.999
    initial_price: float = 0.5


class AgentCreateRequest(BaseModel):
    name: str
    cash: float = Field(100.0, gt=0)
    belief: Optional[float] = Field(default=None, ge=0.01, le=0.99)
    rho: Optional[float] = Field(default=None, ge=0.01, le=10.0)
    personality: Optional[Dict[str, Any]] = None


class AgentPatchRequest(BaseModel):
    name: Optional[str] = None
    cash: Optional[float] = Field(default=None, gt=0)
    belief: Optional[float] = Field(default=None, ge=0.01, le=0.99)
    rho: Optional[float] = Field(default=None, ge=0.01, le=10.0)
    personality: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def at_least_one(self) -> "AgentPatchRequest":
        if (
            self.name is None
            and self.cash is None
            and self.belief is None
            and self.rho is None
            and self.personality is None
        ):
            raise ValueError("Provide at least one field to update")
        return self


class MarketJoinRequest(BaseModel):
    agent_id: int


class LmsrTradeRequest(BaseModel):
    agent_id: int
    quantity: float


class CdaTradeRequest(BaseModel):
    agent_id: int
    side: str
    quantity: float
    limit_price: Optional[float] = None
    order_type: str = "limit"


class BeliefUpdateRequest(BaseModel):
    new_belief: Optional[float] = None
    delta: Optional[float] = None

    @model_validator(mode="after")
    def one_of(self) -> "BeliefUpdateRequest":
        if (self.new_belief is None) == (self.delta is None):
            raise ValueError("Provide exactly one of new_belief or delta")
        if self.new_belief is not None and not (0.01 <= float(self.new_belief) <= 0.99):
            raise ValueError("new_belief must be in [0.01, 0.99]")
        return self


class NewsEventRequest(BaseModel):
    headline: str = "Breaking news"
    new_belief: Optional[float] = None
    delta: Optional[float] = None
    affected_fraction: float = Field(0.5, ge=0.0, le=1.0)
    min_signal_sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    agent_ids: Optional[List[int]] = None

    @model_validator(mode="after")
    def one_of(self) -> "NewsEventRequest":
        if (self.new_belief is None) == (self.delta is None):
            raise ValueError("Provide exactly one of new_belief or delta")
        if self.new_belief is not None and not (0.01 <= float(self.new_belief) <= 0.99):
            raise ValueError("new_belief must be in [0.01, 0.99]")
        return self


def _http_from_value(err: ValueError, *, not_found: bool = False) -> None:
    msg = str(err)
    low = msg.lower()
    if "not found" in low:
        raise HTTPException(status_code=404, detail=msg)
    if not_found:
        raise HTTPException(status_code=404, detail=msg)
    if "status" in low and "trade" in low:
        raise HTTPException(status_code=409, detail=msg)
    raise HTTPException(status_code=400, detail=msg)


def _parse_personality(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _signal_profile(personality: Dict[str, Any]) -> Tuple[float, float]:
    sensitivity = float(personality.get("signal_sensitivity", _DEFAULT_SIGNAL_SENSITIVITY))
    stubbornness = float(personality.get("stubbornness", _DEFAULT_STUBBURNNESS))
    return float(np.clip(sensitivity, 0.0, 1.0)), float(np.clip(stubbornness, 0.0, 1.0))


def _agent_response_row(agent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "agent_id": int(agent["id"]),
        "name": str(agent["name"]),
        "cash": float(agent["cash"]),
        "belief": float(agent["belief"]) if agent.get("belief") is not None else None,
        "rho": float(agent["rho"]) if agent.get("rho") is not None else None,
        "personality": _parse_personality(agent.get("personality")),
        "created_at": agent.get("created_at"),
    }


# --- Routes ---


@agents_router.post("/agents", status_code=201)
def create_agent(body: AgentCreateRequest) -> Dict[str, Any]:
    svc = get_market_service()
    # Sample a personality at agent creation time so every agent has a
    # fully-specified, market-independent personality from birth.
    if body.personality is not None:
        personality_dict = body.personality
    else:
        personality_dict = sample_personality(DEFAULT_POPULATION_DIST).to_dict()
    personality_json = json.dumps(personality_dict)
    try:
        agent = svc.create_agent(
            body.name,
            body.cash,
            belief=body.belief,
            rho=body.rho,
            personality=personality_json,
        )
    except ValueError as e:
        _http_from_value(e)
    get_agent_runner().register_or_update_agent(agent)
    return _agent_response_row(agent)


@agents_router.post("/agents/create", status_code=201)
def create_agent_alias(body: AgentCreateRequest) -> Dict[str, Any]:
    """Backward-compatible alias for clients using /agents/create."""
    return create_agent(body)


@agents_router.get("/agents")
def list_global_agents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    svc = get_market_service()
    data = svc.list_agents(limit=limit, offset=offset)
    return {
        "agents": [_agent_response_row(a) for a in data["agents"]],
        "total": int(data["total"]),
    }


@agents_router.patch("/agents/{agent_id}")
def patch_agent(agent_id: int, body: AgentPatchRequest) -> Dict[str, Any]:
    svc = get_market_service()
    personality_json = json.dumps(body.personality) if body.personality is not None else None
    try:
        updated = svc.update_agent(
            agent_id,
            name=body.name,
            cash=body.cash,
            belief=body.belief,
            rho=body.rho,
            personality=personality_json,
        )
    except ValueError as e:
        _http_from_value(e, not_found=True)
    get_agent_runner().register_or_update_agent(updated)
    return _agent_response_row(updated)


@router.post("/create", status_code=201)
def create_market(body: MarketCreateRequest) -> Dict[str, Any]:
    """
    Create one market only (no agent creation) and set status to **open**.

    Agents are managed independently through `/api/agents*` endpoints.
    """
    if body.mechanism not in ("lmsr", "cda"):
        raise HTTPException(status_code=400, detail="mechanism must be 'lmsr' or 'cda'")

    svc = get_market_service()
    slug = f"m-{uuid.uuid4().hex[:12]}"

    mkt = svc.create_market(
        slug,
        body.title,
        mechanism=body.mechanism,
        b=body.b,
        ground_truth=body.ground_truth,
        description=body.description,
        tick_size=body.tick_size,
        min_price=body.min_price,
        max_price=body.max_price,
        initial_price=body.initial_price,
    )
    mid = int(mkt["id"])
    _market_initial_cash[mid] = 100.0

    svc.set_market_status(mid, "open")
    price0 = svc.get_price(mid)
    return {
        "market_id": mid,
        "mechanism": body.mechanism,
        "initial_price": price0,
        "ground_truth": body.ground_truth,
        "status": "open",
    }


@router.get("/{market_id}/price")
def get_market_price(market_id: int) -> Dict[str, Any]:
    """Latest mid / LMSR price plus optional CDA quotes and timestamps."""
    svc = get_market_service()
    try:
        snap = svc.get_price_snapshot(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)

    trades = svc.get_trades(market_id=market_id, limit=1)
    last_at = None
    last_px = snap.get("last_trade_price")
    if trades:
        last_at = trades[0].get("created_at")
        if snap.get("mechanism") == "lmsr":
            # LMSR: use price_after from latest trade if you want last print
            last_px = trades[0].get("price_after")

    ts = datetime.now(timezone.utc).isoformat()
    if snap.get("mechanism") == "lmsr":
        return {
            "market_id": market_id,
            "price": float(snap["price"]),
            "best_bid": None,
            "best_ask": None,
            "last_trade_price": last_px,
            "last_trade_at": last_at,
            "timestamp": ts,
        }
    return {
        "market_id": market_id,
        "price": float(snap["price"]),
        "best_bid": snap.get("best_bid"),
        "best_ask": snap.get("best_ask"),
        "last_trade_price": snap.get("last_trade_price"),
        "last_trade_at": last_at,
        "timestamp": ts,
    }


@router.get("/{market_id}/book")
def get_order_book(market_id: int) -> Dict[str, Any]:
    svc = get_market_service()
    try:
        m = svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    if m["mechanism"] != "cda":
        raise HTTPException(status_code=400, detail="order book only for CDA markets")
    ob = svc.get_order_book(market_id)
    return {"bids": ob["bids"], "asks": ob["asks"]}


@router.post("/{market_id}/trade")
def post_trade(market_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """LMSR: ``{agent_id, quantity}``. CDA: ``agent_id, side, quantity, order_type, limit_price?``."""
    svc = get_market_service()
    try:
        m = svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)

    if m["mechanism"] == "lmsr":
        req = LmsrTradeRequest.model_validate(payload)
        try:
            r = svc.execute_lmsr_trade(market_id, req.agent_id, req.quantity)
        except ValueError as e:
            _http_from_value(e)
        if r.get("trade_id") is None:
            raise HTTPException(status_code=400, detail="trade clipped to zero (insufficient funds?)")
        ag = svc.get_agent(req.agent_id, market_id)
        pos = svc.get_position(req.agent_id, market_id)
        ic = _market_initial_cash.get(market_id, 100.0)
        price = float(r["price_after"])
        pnl = float(ag["cash"]) + float(pos["yes_shares"]) * price - ic
        return {
            "trade_id": r["trade_id"],
            "executed_quantity": r["quantity"],
            "executed_price": price,
            "cost": r["cost"],
            "new_price": price,
            "agent_cash_after": float(ag["cash"]),
            "agent_shares_after": float(pos["yes_shares"]),
            "pnl_mark": pnl,
        }

    req = CdaTradeRequest.model_validate(payload)
    try:
        r = svc.execute_cda_order(
            market_id,
            req.agent_id,
            req.side,
            req.quantity,
            req.limit_price,
            req.order_type,
        )
    except ValueError as e:
        _http_from_value(e)
    # Aggregate first persisted trade if any
    px = float(r["price_after"])
    ag = svc.get_agent(req.agent_id, market_id)
    pos = svc.get_position(req.agent_id, market_id)
    ic = _market_initial_cash.get(market_id, 100.0)
    pnl = float(ag["cash"]) + float(pos["yes_shares"]) * px - ic
    tid = r["trades"][0]["trade_id"] if r.get("trades") else None
    return {
        "trade_id": tid,
        "executed_quantity": r["filled_quantity"],
        "executed_price": px,
        "cost": 0.0,
        "new_price": px,
        "agent_cash_after": float(ag["cash"]),
        "agent_shares_after": float(pos["yes_shares"]),
        "pnl_mark": pnl,
        "raw": r,
    }


@router.get("/{market_id}/agent/{agent_id}")
def get_one_agent(market_id: int, agent_id: int) -> Dict[str, Any]:
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    try:
        ag = svc.get_agent(agent_id, market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    pos = svc.get_position(agent_id, market_id)
    price = svc.get_price(market_id)
    ic = _market_initial_cash.get(market_id, 100.0)
    shares = float(pos.get("yes_shares") or 0.0)
    pnl = float(ag["cash"]) + shares * float(price) - ic
    # Belief and personality come from the global agent record, not the position,
    # so that agent personalities remain independent of the markets they trade in.
    parsed: Any = _parse_personality(ag.get("personality"))
    return {
        "agent_id": agent_id,
        "cash": float(ag["cash"]),
        "shares": shares,
        "belief": float(ag["belief"]) if ag.get("belief") is not None else 0.5,
        "rho": float(ag["rho"]) if ag.get("rho") is not None else 1.0,
        "pnl": pnl,
        "personality": parsed,
    }


@router.post("/{market_id}/agent/{agent_id}/belief")
def post_agent_belief(
    market_id: int, agent_id: int, body: BeliefUpdateRequest,
) -> Dict[str, Any]:
    """Set absolute belief or apply a delta (clipped to [0.01, 0.99])."""
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    try:
        svc.get_agent(agent_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    pos = svc.get_position(agent_id, market_id)
    old = float(pos["belief"]) if pos.get("belief") is not None else 0.5
    if body.new_belief is not None:
        new_b = float(np.clip(float(body.new_belief), 0.01, 0.99))
    else:
        assert body.delta is not None
        new_b = float(np.clip(old + float(body.delta), 0.01, 0.99))
    try:
        svc.set_agent_belief(market_id, agent_id, new_b)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    return {
        "agent_id": agent_id,
        "old_belief": old,
        "new_belief": new_b,
        "at_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/{market_id}/join")
def join_market(market_id: int, body: MarketJoinRequest) -> Dict[str, Any]:
    """
    Explicitly ensure a position row for an agent in a market.

    This is optional; trading already lazy-creates positions.
    """
    svc = get_market_service()
    try:
        svc.get_market(market_id)
        svc.get_agent(body.agent_id)
        pos = svc.ensure_position(body.agent_id, market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    return {
        "status": "joined",
        "market_id": market_id,
        "agent_id": body.agent_id,
        "shares": float(pos.get("yes_shares") or 0.0),
    }


@router.get("/{market_id}/agents")
def list_market_agents(
    market_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    rows = svc.list_agents(limit=1_000_000, offset=0)["agents"]
    price = svc.get_price(market_id)
    ic = _market_initial_cash.get(market_id, 100.0)
    out: List[Dict[str, Any]] = []
    for r in rows[offset : offset + limit]:
        aid = int(r["id"])
        pos = svc.get_position(aid, market_id)
        sh = float(pos.get("yes_shares") or 0)
        cash = float(r["cash"])
        bel = float(r["belief"]) if r.get("belief") is not None else 0.5
        rho = float(r["rho"]) if r.get("rho") is not None else 1.0
        pers = _parse_personality(r.get("personality"))
        out.append(
            {
                "agent_id": aid,
                "cash": cash,
                "shares": sh,
                "belief": bel,
                "rho": rho,
                "pnl": cash + sh * float(price) - ic,
                "personality": pers,
            }
        )
    return {"agents": out, "total": len(rows)}


@router.get("/{market_id}/trades")
def list_trades(
    market_id: int,
    since: Optional[int] = Query(None, description="Only trades with id > since"),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    rows = svc.get_trades(market_id=market_id, since_trade_id=since, limit=limit)
    # Oldest-first for feed readability (store returns DESC)
    rows = list(reversed(rows))
    trades_out: List[Dict[str, Any]] = []
    for t in rows:
        trades_out.append(
            {
                "trade_id": str(t["id"]),
                "agent_id": t["agent_id"],
                "quantity": t["shares"],
                "price": float(t["price_after"]),
                "at": t.get("created_at"),
            }
        )
    total = svc.count_trades(market_id)
    return {"trades": trades_out, "total": total}


@router.post("/{market_id}/news")
def inject_news_event(market_id: int, body: NewsEventRequest) -> Dict[str, Any]:
    """
    Apply a persistent belief shock to a subset of agents in this market.

    Selection defaults to the most signal-sensitive agents (descending by
    ``signal_sensitivity * (1 - stubbornness)``) with optional explicit
    ``agent_ids`` override.
    """
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)

    rows = svc.list_agents(limit=1_000_000, offset=0)["agents"]
    by_id = {int(r["id"]): r for r in rows}

    selected: List[Dict[str, Any]] = []
    if body.agent_ids is not None:
        seen: set[int] = set()
        missing: List[int] = []
        for aid in body.agent_ids:
            x = int(aid)
            if x in seen:
                continue
            seen.add(x)
            row = by_id.get(x)
            if row is None:
                missing.append(x)
            else:
                selected.append(row)
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"agent ids not found in market {market_id}: {missing}",
            )
    else:
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for row in rows:
            aid = int(row["id"])
            personality = _parse_personality(row.get("personality"))
            sensitivity, stubbornness = _signal_profile(personality)
            if sensitivity + 1e-12 < body.min_signal_sensitivity:
                continue
            score = sensitivity * (1.0 - stubbornness)
            scored.append((score, aid, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        n_pick = int(round(len(scored) * body.affected_fraction))
        if body.affected_fraction > 0 and n_pick == 0 and scored:
            n_pick = 1
        if body.affected_fraction == 0:
            n_pick = 0
        selected = [row for _, _, row in scored[:n_pick]]

    changed: List[Dict[str, Any]] = []
    for row in selected:
        aid = int(row["id"])
        old_belief = float(row["belief"]) if row.get("belief") is not None else 0.5
        personality = _parse_personality(row.get("personality"))
        sensitivity, stubbornness = _signal_profile(personality)
        influence = float(np.clip(sensitivity * (1.0 - stubbornness), 0.0, 1.0))
        target = float(body.new_belief) if body.new_belief is not None else old_belief + float(body.delta)
        raw_new = old_belief + (target - old_belief) * influence
        new_belief = float(np.clip(raw_new, 0.01, 0.99))
        try:
            svc.set_agent_belief(market_id, aid, new_belief)
        except ValueError as e:
            _http_from_value(e, not_found=True)
        changed.append(
            {
                "agent_id": aid,
                "old_belief": old_belief,
                "new_belief": new_belief,
                "signal_sensitivity": sensitivity,
                "stubbornness": stubbornness,
                "influence": influence,
            }
        )

    if changed:
        mean_before = float(np.mean([r["old_belief"] for r in changed]))
        mean_after = float(np.mean([r["new_belief"] for r in changed]))
    else:
        mean_before = 0.0
        mean_after = 0.0

    return {
        "market_id": market_id,
        "headline": body.headline,
        "mode": "absolute" if body.new_belief is not None else "delta",
        "requested_new_belief": body.new_belief,
        "requested_delta": body.delta,
        "affected_fraction": body.affected_fraction,
        "min_signal_sensitivity": body.min_signal_sensitivity,
        "n_candidates": len(rows),
        "n_affected": len(changed),
        "mean_belief_before": mean_before,
        "mean_belief_after": mean_after,
        "affected_agents": changed,
        "at_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/{market_id}/start")
def start_autonomous(market_id: int) -> Dict[str, Any]:
    """Start autonomous trading lifecycle for one market."""
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)

    runner = get_agent_runner()
    if runner.is_running(market_id):
        raise HTTPException(status_code=409, detail="market already running")
    try:
        n_active = runner.start_market(market_id)
    except ValueError as e:
        _http_from_value(e)

    return {"status": "started", "n_agents_running": n_active}


@router.post("/{market_id}/stop")
def stop_autonomous(market_id: int) -> Dict[str, Any]:
    """Stop autonomous lifecycle for one market with thread cleanup."""
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)

    runner = get_agent_runner()
    if not runner.is_running(market_id):
        raise HTTPException(status_code=409, detail="market is not running")
    try:
        stop_info = runner.stop_market(market_id)
    except ValueError as e:
        _http_from_value(e)

    n = len(svc.get_trades(market_id=market_id, limit=100_000))
    return {
        "status": "stopped",
        "total_trades": n,
        "duration_sec": stop_info["duration_sec"],
        "zombie_threads": stop_info["zombie_threads"],
    }
