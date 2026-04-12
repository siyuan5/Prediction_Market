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
import threading
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

from belief_init import BeliefSpec, sample_beliefs  # noqa: E402
from market_service import MarketService  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])

# --- Singleton service (reset in tests via ``reset_market_runtime``) ---
_market_service: Optional[MarketService] = None
# market_id -> metadata from create()
_market_initial_cash: Dict[int, float] = {}
# market_id -> list of (thread, AutonomousAgent) for stop()
_autonomous_runners: Dict[int, List[Tuple[threading.Thread, Any]]] = {}
_runners_lock = threading.Lock()


def reset_market_runtime() -> None:
    """Clear singleton and runners (used by tests only)."""
    global _market_service
    _market_service = None
    _market_initial_cash.clear()
    with _runners_lock:
        for mid, pairs in list(_autonomous_runners.items()):
            for _th, ag in pairs:
                try:
                    ag.stop()
                except Exception:
                    pass
        _autonomous_runners.clear()


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


# --- Pydantic request bodies (match FINAL_PHASE_TASKS contracts) ---


class RhoDistribution(BaseModel):
    min: float = Field(0.5, ge=0.01, le=10.0)
    max: float = Field(2.0, ge=0.01, le=10.0)


class BeliefSpecPayload(BaseModel):
    mode: str = "gaussian"
    sigma: float = Field(0.10, ge=0.0, le=0.5)


class MarketCreateRequest(BaseModel):
    mechanism: str = "lmsr"
    ground_truth: float = Field(0.70, ge=0.01, le=0.99)
    n_agents: int = Field(50, ge=2, le=500)
    initial_cash: float = Field(100.0, gt=0)
    b: float = Field(100.0, gt=0)
    belief_spec: BeliefSpecPayload = Field(default_factory=BeliefSpecPayload)
    rho_distribution: RhoDistribution = Field(default_factory=RhoDistribution)
    personality_defaults: Optional[Dict[str, float]] = None
    seed: int = 42
    title: str = "Autonomous market"


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


# --- Routes ---


@router.post("/create", status_code=201)
def create_market(body: MarketCreateRequest) -> Dict[str, Any]:
    """
    Create one market, ``n_agents`` agents with sampled beliefs and rhos, then
    set status to **open** so manual or autonomous trades are allowed.
    """
    if body.mechanism not in ("lmsr", "cda"):
        raise HTTPException(status_code=400, detail="mechanism must be 'lmsr' or 'cda'")
    if body.rho_distribution.min > body.rho_distribution.max:
        raise HTTPException(status_code=400, detail="rho_distribution.min must be <= max")

    svc = get_market_service()
    rng = np.random.default_rng(body.seed)
    slug = f"m-{uuid.uuid4().hex[:12]}"
    spec = BeliefSpec(
        mode=body.belief_spec.mode,
        sigma=body.belief_spec.sigma,
    )
    beliefs = sample_beliefs(
        body.ground_truth, body.n_agents, spec, rng,
    )
    pers_json = json.dumps(body.personality_defaults or {})

    mkt = svc.create_market(
        slug,
        body.title,
        mechanism=body.mechanism,
        b=body.b,
        ground_truth=body.ground_truth,
    )
    mid = int(mkt["id"])
    _market_initial_cash[mid] = float(body.initial_cash)

    for i in range(body.n_agents):
        rho = float(rng.uniform(body.rho_distribution.min, body.rho_distribution.max))
        svc.create_agent(
            f"auto-{mid}-{i}",
            body.initial_cash,
            market_id=mid,
            belief=float(beliefs[i]),
            rho=rho,
            personality=pers_json,
        )

    svc.set_market_status(mid, "open")
    price0 = svc.get_price(mid)
    return {
        "market_id": mid,
        "mechanism": body.mechanism,
        "n_agents": body.n_agents,
        "initial_price": price0,
        "ground_truth": body.ground_truth,
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
    pers = pos.get("personality")
    parsed: Any = pers
    if isinstance(pers, str) and pers.strip().startswith("{"):
        try:
            parsed = json.loads(pers)
        except json.JSONDecodeError:
            pass
    return {
        "agent_id": agent_id,
        "cash": float(ag["cash"]),
        "shares": shares,
        "belief": float(pos["belief"]) if pos.get("belief") is not None else 0.5,
        "rho": float(pos["rho"]) if pos.get("rho") is not None else 1.0,
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
    pos = svc.get_position(agent_id, market_id)
    if pos.get("belief") is None:
        raise HTTPException(status_code=404, detail="agent has no position in this market")
    old = float(pos["belief"])
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


@router.get("/{market_id}/agents")
def list_agents(
    market_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)
    rows = svc.list_agents_for_market(market_id)
    price = svc.get_price(market_id)
    ic = _market_initial_cash.get(market_id, 100.0)
    out: List[Dict[str, Any]] = []
    for r in rows[offset : offset + limit]:
        aid = int(r["agent_id"])
        sh = float(r["yes_shares"] or 0)
        cash = float(r["cash"])
        bel = float(r["belief"]) if r.get("belief") is not None else 0.5
        rho = float(r["rho"]) if r.get("rho") is not None else 1.0
        out.append(
            {
                "agent_id": aid,
                "cash": cash,
                "shares": sh,
                "belief": bel,
                "rho": rho,
                "pnl": cash + sh * float(price) - ic,
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


def _run_autonomous_loop(agent: Any) -> None:
    """Thread target: run until ``agent.stop()``."""
    try:
        agent.run()
    except Exception:
        logger.exception("Autonomous agent thread crashed")


@router.post("/{market_id}/start")
def start_autonomous(market_id: int) -> Dict[str, Any]:
    """Spawn one polling thread per agent in this market (LMSR-oriented)."""
    from autonomous_agent import AutonomousAgent  # noqa: WPS433 — lazy import

    svc = get_market_service()
    try:
        m = svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)

    with _runners_lock:
        if market_id in _autonomous_runners and _autonomous_runners[market_id]:
            raise HTTPException(status_code=409, detail="autonomous runners already active")

    try:
        svc.set_market_status(market_id, "running")
    except ValueError as e:
        _http_from_value(e)

    rows = svc.list_agents_for_market(market_id)
    base = _autonomous_api_base()
    threads: List[Tuple[threading.Thread, Any]] = []
    for r in rows:
        aid = int(r["agent_id"])
        pers = r.get("personality")
        pmap: Optional[Dict[str, Any]] = None
        if isinstance(pers, str) and pers.strip():
            try:
                pmap = json.loads(pers)
            except json.JSONDecodeError:
                pmap = None
        ag = AutonomousAgent(
            agent_id=aid,
            market_id=str(market_id),
            api_base_url=base,
            personality=pmap,
        )
        th = threading.Thread(target=_run_autonomous_loop, args=(ag,), daemon=True)
        th.start()
        threads.append((th, ag))

    with _runners_lock:
        _autonomous_runners[market_id] = threads

    return {"status": "started", "n_agents_running": len(threads)}


@router.post("/{market_id}/stop")
def stop_autonomous(market_id: int) -> Dict[str, Any]:
    """Signal all autonomous agents to stop and set market status to **stopped**."""
    svc = get_market_service()
    try:
        svc.get_market(market_id)
    except ValueError as e:
        _http_from_value(e, not_found=True)

    with _runners_lock:
        pairs = _autonomous_runners.pop(market_id, [])

    for _th, ag in pairs:
        try:
            ag.stop()
        except Exception:
            pass

    try:
        svc.set_market_status(market_id, "stopped")
    except ValueError:
        pass

    n = len(svc.get_trades(market_id=market_id, limit=100_000))
    return {"status": "stopped", "total_trades": n, "duration_sec": 0.0}
