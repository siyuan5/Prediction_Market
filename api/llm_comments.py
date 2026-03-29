"""
Event-aware agent comments via local Ollama (OpenAI-compatible /api/chat).

Environment (optional):
  COMMENT_USE_LLM     — "0" / "false" / "no" to disable and use templates only.
  OLLAMA_BASE_URL     — default http://127.0.0.1:11434
  OLLAMA_MODEL        — default llama3.2 (pull the same name in Ollama)
  OLLAMA_TIMEOUT      — seconds per request, default 8
  COMMENT_LLM_MAX     — max LLM calls per simulation or session (default 15); counts failures too so runs don’t hang retrying Ollama.
"""

from __future__ import annotations

import json
import os
import re
import random
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from .comments import pick_filler_comment

_SYSTEM = (
    "You are one trader in a prediction market. Reply with a single short line of in-character "
    "chat (max 200 characters). Plain text only: no quotes wrapping the line, no hashtags, no markdown."
)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except ValueError:
        return default


def llm_budget_initial() -> int:
    """Max LLM attempts per /simulate or interactive session (each attempt uses one slot, success or fail)."""
    n = _env_int("COMMENT_LLM_MAX", 15)
    return max(0, n)


def _ollama_chat(user_prompt: str) -> Optional[str]:
    if not _env_bool("COMMENT_USE_LLM", True):
        return None

    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
    timeout = _env_float("OLLAMA_TIMEOUT", 8.0)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.82, "num_predict": 140},
    }
    url = f"{base}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    msg = raw.get("message") if isinstance(raw, dict) else None
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return _sanitize_line(content)


def _sanitize_line(s: str) -> str:
    t = s.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" \t\"'“”")
    if len(t) > 240:
        t = t[:237] + "…"
    return t if t else ""


def _user_prompt(
    *,
    event_name: str,
    mechanism: str,
    belief: float,
    agent_id: int,
    round_num: int,
    market_yes_price: float,
) -> str:
    mech = "LMSR (automated market maker)" if mechanism == "lmsr" else "CDA (order book)"
    return (
        f"Market question: {event_name}\n"
        f"Mechanism: {mech}.\n"
        f"You are trader #{agent_id}. Trading round {round_num}.\n"
        f"Your subjective probability the question resolves YES: {belief * 100:.1f}%.\n"
        f"The market price (implied probability of YES) is about {market_yes_price * 100:.1f}%.\n\n"
        "Write one casual comment about this specific question—your edge, doubt, or what you’re watching. "
        "Stay consistent with whether you lean YES or NO vs 50%."
    )


def generate_comment_text(
    *,
    event_name: str,
    mechanism: str,
    belief: float,
    agent_id: int,
    round_num: int,
    market_yes_price: float,
    rng: random.Random,
    llm_budget: Optional[List[int]],
) -> Tuple[str, str]:
    """
    Returns (text, source) where source is 'llm' or 'template'.
    llm_budget: single-element list [remaining]; decremented on each LLM attempt so failed/timeouts
    cannot repeat unbounded times for every comment slot.
    """
    use_slot = (
        llm_budget is not None
        and len(llm_budget) > 0
        and llm_budget[0] > 0
        and _env_bool("COMMENT_USE_LLM", True)
    )

    if use_slot:
        llm_budget[0] -= 1
        prompt = _user_prompt(
            event_name=event_name,
            mechanism=mechanism,
            belief=belief,
            agent_id=agent_id,
            round_num=round_num,
            market_yes_price=market_yes_price,
        )
        out = _ollama_chat(prompt)
        if out:
            return out, "llm"

    return (
        pick_filler_comment(belief, agent_id, round_num, rng),
        "template",
    )
