"""
Real-time intra-round profitability tracking for prediction market simulations.

Supports streaming snapshots during simulation (round-by-round) and post-hoc analysis.
Works with both LMSR (Team A) and CDA (Team B) mechanisms.

Profitability is computed as:
- Realized P&L: cash change from trades
- Mark-to-market P&L: (shares * current_price) - (shares * entry_price)
- Terminal P&L: final wealth - initial_cash (post-settlement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class AgentSnapshot:
    """Per-agent profitability state at a given round."""
    agent_id: int
    round_num: int
    belief: float
    rho: float
    
    # Portfolio state
    cash: float
    shares: float
    market_price: float
    
    # Profitability metrics
    initial_cash: float
    entry_price: float  # avg price at which shares were acquired
    
    # Realized profit (from trades)
    realized_pnl: float
    
    # Mark-to-market profit (unrealized, based on current price)
    unrealized_pnl: float
    
    # Total portfolio value
    portfolio_value: float  # cash + shares * market_price
    
    # Cumulative profit since start
    total_pnl: float  # portfolio_value - initial_cash
    
    # Trade activity this round
    trades_this_round: int = 0
    net_shares_traded: float = 0.0
    

@dataclass
class RoundSnapshot:
    """Market-level profitability snapshot for a given round."""
    round_num: int
    market_price: float
    ground_truth: float
    signal: Optional[float] = None
    
    # Trading activity
    total_volume: float = 0.0
    num_trades: int = 0
    
    # Agent aggregates
    agent_snapshots: List[AgentSnapshot] = field(default_factory=list)
    
    # Market-level profitability stats
    avg_profit: float = 0.0  # mean total_pnl across agents
    median_profit: float = 0.0
    std_profit: float = 0.0
    
    # Inequality & concentration
    gini_coefficient: float = 0.0  # profit concentration
    max_profit: float = 0.0
    min_profit: float = 0.0
    
    # Belief accuracy impact
    avg_belief_error: float = 0.0  # mean |belief - ground_truth|
    belief_accuracy_premium: float = 0.0  # correlation coefficient
    

class AgentProfitabilityTracker:
    """
    Tracks per-agent profitability metrics in real-time.
    
    Maintains:
    - Portfolio state (cash, shares, prices)
    - Trade history for realized P&L computation
    - Entry prices for mark-to-market calculations
    """
    
    def __init__(
        self,
        agent_id: int,
        initial_cash: float,
        belief: float,
        rho: float,
    ):
        self.agent_id = agent_id
        self.initial_cash = initial_cash
        self.belief = belief
        self.rho = rho
        
        # Current state
        self.cash = initial_cash
        self.shares = 0.0
        self.last_price = 0.5  # assume fair price at start
        self.entry_price = 0.5  # weighted avg entry price
        self.realized_pnl = 0.0  # profit from trades already settled
        
        # Trade tracking
        self.trade_history: List[Dict[str, float]] = []
        self.shares_acquired_at: Dict[float, float] = {}  # price -> quantity mapping
        
    def update_belief(self, new_belief: float) -> None:
        """Update agent's belief (used in Phase 2)."""
        self.belief = new_belief
        
    def record_trade(
        self,
        trade_qty: float,
        trade_price: float,
        round_num: int,
    ) -> None:
        """
        Record a trade execution.
        
        Args:
            trade_qty: shares traded (positive=buy, negative=sell)
            trade_price: price per share
            round_num: simulation round
        """
        if trade_qty == 0:
            return
            
        # Update cash (no slippage, frictionless)
        self.cash -= trade_qty * trade_price
        
        # Update shares
        old_shares = self.shares
        self.shares += trade_qty
        
        # Track for entry price calculation
        if trade_qty > 0:  # buying
            self.shares_acquired_at[trade_price] = \
                self.shares_acquired_at.get(trade_price, 0.0) + trade_qty
        
        # Recalculate weighted entry price
        self._update_entry_price()
        
        # Record trade
        self.trade_history.append({
            "round": round_num,
            "qty": trade_qty,
            "price": trade_price,
            "cash_before": self.cash + trade_qty * trade_price,
            "shares_before": old_shares,
        })
        
        self.last_price = trade_price
        
    def get_snapshot(
        self,
        round_num: int,
        market_price: float,
    ) -> AgentSnapshot:
        """
        Generate current profitability snapshot.
        
        Args:
            round_num: current simulation round
            market_price: current market price
            
        Returns:
            AgentSnapshot with all profitability metrics
        """
        # Mark-to-market unrealized P&L
        unrealized_pnl = self.shares * (market_price - self.entry_price) if self.shares else 0.0
        
        # Total portfolio value
        portfolio_value = self.cash + self.shares * market_price
        
        # Total P&L from initial cash
        total_pnl = portfolio_value - self.initial_cash
        
        # Count trades this round (simple heuristic: last trade in history)
        trades_this_round = 1 if self.trade_history and \
            self.trade_history[-1].get("round") == round_num else 0
        net_shares_traded = self.trade_history[-1].get("qty", 0.0) if trades_this_round else 0.0
        
        return AgentSnapshot(
            agent_id=self.agent_id,
            round_num=round_num,
            belief=self.belief,
            rho=self.rho,
            cash=self.cash,
            shares=self.shares,
            market_price=market_price,
            initial_cash=self.initial_cash,
            entry_price=self.entry_price,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized_pnl,
            portfolio_value=portfolio_value,
            total_pnl=total_pnl,
            trades_this_round=trades_this_round,
            net_shares_traded=net_shares_traded,
        )
        
    def _update_entry_price(self) -> None:
        """Recompute weighted average entry price from acquired shares."""
        if self.shares <= 0:
            self.entry_price = self.last_price
            return
            
        total_cost = sum(price * qty for price, qty in self.shares_acquired_at.items())
        total_qty = sum(self.shares_acquired_at.values())
        self.entry_price = total_cost / total_qty if total_qty > 0 else self.last_price
        
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Return full trade history for this agent."""
        return self.trade_history.copy()


class MarketProfitabilityAnalyzer:
    """
    Aggregates per-agent snapshots into market-level profitability statistics.
    Computes inequality metrics, belief-accuracy premiums, and trading metrics.
    """
    
    @staticmethod
    def create_round_snapshot(
        round_num: int,
        market_price: float,
        ground_truth: float,
        agent_snapshots: List[AgentSnapshot],
        total_volume: float = 0.0,
        signal: Optional[float] = None,
    ) -> RoundSnapshot:
        """
        Create market-level snapshot from agent snapshots.
        
        Args:
            round_num: simulation round
            market_price: current market price
            ground_truth: true outcome probability
            agent_snapshots: list of per-agent snapshots
            total_volume: total shares traded this round
            signal: public signal (if Phase 2)
            
        Returns:
            RoundSnapshot with aggregated metrics
        """
        snapshot = RoundSnapshot(
            round_num=round_num,
            market_price=market_price,
            ground_truth=ground_truth,
            signal=signal,
            total_volume=total_volume,
            num_trades=len(agent_snapshots),  # approx
            agent_snapshots=agent_snapshots,
        )
        
        if agent_snapshots:
            profits = [s.total_pnl for s in agent_snapshots]
            beliefs = [s.belief for s in agent_snapshots]
            
            # Profit statistics
            snapshot.avg_profit = float(np.mean(profits))
            snapshot.median_profit = float(np.median(profits))
            snapshot.std_profit = float(np.std(profits)) if len(profits) > 1 else 0.0
            snapshot.max_profit = float(np.max(profits))
            snapshot.min_profit = float(np.min(profits))
            
            # Gini coefficient (profit concentration)
            snapshot.gini_coefficient = MarketProfitabilityAnalyzer._gini(profits)
            
            # Belief accuracy (absolute error vs ground truth)
            belief_errors = [abs(b - ground_truth) for b in beliefs]
            snapshot.avg_belief_error = float(np.mean(belief_errors))
            
            # Correlation: belief accuracy vs profit (higher belief accuracy → higher profit?)
            if len(belief_errors) > 1 and len(profits) > 1:
                try:
                    corr = float(np.corrcoef([-e for e in belief_errors], profits)[0, 1])
                    snapshot.belief_accuracy_premium = corr
                except (ValueError, RuntimeWarning):
                    snapshot.belief_accuracy_premium = 0.0
        
        return snapshot
    
    @staticmethod
    def _gini(values: List[float]) -> float:
        """
        Compute Gini coefficient (0=perfect equality, 1=perfect inequality).
        
        For profits, higher Gini means more concentrated winners/losers.
        """
        if len(values) < 2:
            return 0.0
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        
        # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        numerator = 2.0 * np.sum(np.arange(1, n+1) * sorted_vals)
        denominator = n * np.sum(sorted_vals)
        
        if denominator == 0:
            return 0.0
        
        gini = numerator / denominator - (n + 1) / n
        return float(np.clip(gini, 0.0, 1.0))


class ProfitabilitySession:
    """
    Manages profitability tracking across entire simulation.
    
    Accumulates snapshots round-by-round for both real-time streaming
    and post-hoc analysis.
    """
    
    def __init__(self, simulation_id: str = "default"):
        self.simulation_id = simulation_id
        self.round_snapshots: List[RoundSnapshot] = []
        self.agent_trackers: Dict[int, AgentProfitabilityTracker] = {}
        
    def register_agent(
        self,
        agent_id: int,
        initial_cash: float,
        belief: float,
        rho: float,
    ) -> None:
        """Register an agent for profitability tracking."""
        self.agent_trackers[agent_id] = AgentProfitabilityTracker(
            agent_id=agent_id,
            initial_cash=initial_cash,
            belief=belief,
            rho=rho,
        )
        
    def update_agent_belief(self, agent_id: int, new_belief: float) -> None:
        """Update agent belief (Phase 2 signal updates)."""
        if agent_id in self.agent_trackers:
            self.agent_trackers[agent_id].update_belief(new_belief)
            
    def record_trade(
        self,
        agent_id: int,
        trade_qty: float,
        trade_price: float,
        round_num: int,
    ) -> None:
        """Record a trade execution."""
        if agent_id in self.agent_trackers:
            self.agent_trackers[agent_id].record_trade(
                trade_qty=trade_qty,
                trade_price=trade_price,
                round_num=round_num,
            )
            
    def snapshot_round(
        self,
        round_num: int,
        market_price: float,
        ground_truth: float,
        total_volume: float = 0.0,
        signal: Optional[float] = None,
    ) -> RoundSnapshot:
        """
        Capture profitability snapshot for current round.
        
        Args:
            round_num: simulation round number
            market_price: current market price
            ground_truth: true outcome probability
            total_volume: total volume traded this round
            signal: public signal (if Phase 2)
            
        Returns:
            RoundSnapshot (also stored internally)
        """
        agent_snapshots = [
            tracker.get_snapshot(round_num, market_price)
            for tracker in self.agent_trackers.values()
        ]
        
        round_snapshot = MarketProfitabilityAnalyzer.create_round_snapshot(
            round_num=round_num,
            market_price=market_price,
            ground_truth=ground_truth,
            agent_snapshots=agent_snapshots,
            total_volume=total_volume,
            signal=signal,
        )
        
        self.round_snapshots.append(round_snapshot)
        return round_snapshot
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics across all rounds.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.round_snapshots:
            return {}
        
        # Extract final round data
        final_round = self.round_snapshots[-1]
        
        # Profit distribution over time
        all_profits = [
            [s.total_pnl for s in rs.agent_snapshots]
            for rs in self.round_snapshots
        ]
        
        return {
            "simulation_id": self.simulation_id,
            "total_rounds": len(self.round_snapshots),
            "num_agents": len(self.agent_trackers),
            "final_market_price": final_round.market_price,
            "final_avg_profit": final_round.avg_profit,
            "final_max_profit": final_round.max_profit,
            "final_min_profit": final_round.min_profit,
            "final_gini": final_round.gini_coefficient,
            "avg_total_volume": float(np.mean([r.total_volume for r in self.round_snapshots])),
            "overall_belief_accuracy_premium": final_round.belief_accuracy_premium,
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Export full session data as dictionary."""
        return {
            "simulation_id": self.simulation_id,
            "round_snapshots": [
                {
                    "round_num": rs.round_num,
                    "market_price": rs.market_price,
                    "ground_truth": rs.ground_truth,
                    "signal": rs.signal,
                    "total_volume": rs.total_volume,
                    "num_trades": rs.num_trades,
                    "avg_profit": rs.avg_profit,
                    "median_profit": rs.median_profit,
                    "std_profit": rs.std_profit,
                    "max_profit": rs.max_profit,
                    "min_profit": rs.min_profit,
                    "gini_coefficient": rs.gini_coefficient,
                    "avg_belief_error": rs.avg_belief_error,
                    "belief_accuracy_premium": rs.belief_accuracy_premium,
                    "agent_snapshots": [
                        {
                            "agent_id": a.agent_id,
                            "belief": a.belief,
                            "rho": a.rho,
                            "cash": a.cash,
                            "shares": a.shares,
                            "market_price": a.market_price,
                            "entry_price": a.entry_price,
                            "realized_pnl": a.realized_pnl,
                            "unrealized_pnl": a.unrealized_pnl,
                            "portfolio_value": a.portfolio_value,
                            "total_pnl": a.total_pnl,
                            "trades_this_round": a.trades_this_round,
                            "net_shares_traded": a.net_shares_traded,
                        }
                        for a in rs.agent_snapshots
                    ],
                }
                for rs in self.round_snapshots
            ],
            "summary": self.get_summary(),
        }
