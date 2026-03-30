"""
Template-only trader chat lines when Ollama is off, the budget is exhausted, or the LLM fails.

Pools are chosen by coarse belief buckets so wording matches YES vs NO lean.
"""

from __future__ import annotations

import random
from typing import List


def _pool_for_belief(belief: float) -> List[str]:
    # belief = P(Yes). Wording should match whether the agent leans No vs Yes.
    if belief < 0.38:
        return [
            "I’m priced on the No side—don’t see the path to Yes.",
            "This resolves No unless something big shifts.",
            "Selling the rumor; Yes looks like a long shot.",
            "My book is mostly No here.",
            "Would need a surprise to flip this to Yes.",
        ]
    if belief < 0.48:
        return [
            "Slight edge to No for me still.",
            "Under 50% on Yes—cautious.",
            "Leaning against Yes until we see more signal.",
            "No is still the cleaner story.",
            "Not ready to buy Yes at these levels.",
        ]
    if belief < 0.52:
        return [
            "Basically a coin flip—could go either way.",
            "No strong lean; watching the next prints.",
            "Market and my prior disagree a bit—waiting.",
            "Could nudge either direction next round.",
            "Flat conviction; staying small.",
        ]
    if belief < 0.65:
        return [
            "Slight lean to Yes, nothing heroic.",
            "Yes a bit under my fair, but not a slam dunk.",
            "Small long Yes—size is modest.",
            "Edge to Yes, still plenty of No risk.",
            "Cautiously bidding Yes here.",
        ]
    if belief < 0.82:
        return [
            "Leaning Yes—odds look reasonable.",
            "I like the risk/reward on Yes from here.",
            "Momentum seems to support the Yes side.",
            "Fair value feels north of where we trade.",
            "Willing to add on Yes on weakness.",
        ]
    return [
        "High conviction this resolves Yes.",
        "Yes side looks cheap to me.",
        "This is a core Yes for my book.",
        "I’d be shocked if this ends No.",
        "All-in tone: I’m very long Yes here.",
    ]


def pick_filler_comment(
    belief: float,
    agent_id: int,
    round_num: int,
    rng: random.Random,
) -> str:
    """Deterministic-ish variety: same seed + ids → stable text for a given comment event."""
    pool = _pool_for_belief(belief)
    idx = (agent_id * 1_000_003 + round_num * 97 + rng.randint(0, 10_000)) % len(pool)
    return pool[idx]
