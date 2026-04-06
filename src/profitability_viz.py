"""
Profitability visualization utilities for matplotlib/seaborn.

Generates comprehensive visual analyses:
- Profit timeseries by agent and aggregate
- Belief accuracy vs profit scatter plots
- Rho-based aggregates
- Inequality metrics over time
- Distribution analysis
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from .profitability_analysis import ProfitabilitySession, RoundSnapshot, AgentSnapshot
except ImportError:
    from profitability_analysis import ProfitabilitySession, RoundSnapshot, AgentSnapshot


def _require_matplotlib(func_name: str) -> None:
    """Raise helpful error if matplotlib is not available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            f"matplotlib and seaborn are required for {func_name}.\n"
            "Install with: pip install matplotlib seaborn"
        )


class ProfitabilityVisualizer:
    """High-level API for generating profitability visualizations."""
    
    def __init__(
        self,
        session: ProfitabilitySession,
        style: str = "darkgrid",
        figsize: Tuple[int, int] = (14, 8),
    ):
        """
        Initialize visualizer.
        
        Args:
            session: ProfitabilitySession with snapshot data
            style: seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
            figsize: default figure size (width, height)
        """
        _require_matplotlib("ProfitabilityVisualizer")
        
        self.session = session
        self.style = style
        self.figsize = figsize
        
        sns.set_style(style)
        sns.set_palette("husl")
    
    def plot_profit_timeseries(
        self,
        aggregate_only: bool = False,
        output_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot profit timeseries: all agents or market aggregates.
        
        Args:
            aggregate_only: if True, show only mean/median/std bands
            output_path: if provided, save figure to this path
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if aggregate_only:
            # Plot market aggregates
            rounds = [rs.round_num for rs in self.session.round_snapshots]
            avg_profits = [rs.avg_profit for rs in self.session.round_snapshots]
            median_profits = [rs.median_profit for rs in self.session.round_snapshots]
            std_profits = [rs.std_profit for rs in self.session.round_snapshots]
            
            ax.plot(rounds, avg_profits, label="Mean Profit", linewidth=2.5, marker='o')
            ax.plot(rounds, median_profits, label="Median Profit", linewidth=2.5, marker='s')
            
            # Confidence band
            upper = np.array(avg_profits) + np.array(std_profits)
            lower = np.array(avg_profits) - np.array(std_profits)
            ax.fill_between(rounds, lower, upper, alpha=0.2, label="±1 Std Dev")
            
            ax.set_ylabel("Profit ($)", fontsize=12)
            ax.set_title("Market-Level Profit Timeseries", fontsize=14, fontweight='bold')
        else:
            # Plot per-agent
            agent_ids = list(self.session.agent_trackers.keys())
            colors = sns.color_palette("husl", len(agent_ids))
            
            for agent_id, color in zip(agent_ids, colors):
                profits = []
                for rs in self.session.round_snapshots:
                    snap = next((s for s in rs.agent_snapshots if s.agent_id == agent_id), None)
                    if snap:
                        profits.append(snap.total_pnl)
                
                if profits:
                    rounds = [rs.round_num for rs in self.session.round_snapshots][:len(profits)]
                    ax.plot(rounds, profits, label=f"Agent {agent_id}", alpha=0.7, linewidth=1.5)
            
            ax.set_ylabel("Profit ($)", fontsize=12)
            ax.set_title("Individual Agent Profit Timeseries", fontsize=14, fontweight='bold')
        
        ax.set_xlabel("Round", fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_belief_accuracy_vs_profit(
        self,
        round_num: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Figure:
        """
        Scatter plot: belief accuracy (vs ground truth) vs profit.
        
        Args:
            round_num: if specified, show only this round; else use final round
            output_path: if provided, save figure to this path
            
        Returns:
            matplotlib Figure
        """
        if not self.session.round_snapshots:
            raise ValueError("No data to plot")
        
        if round_num is None:
            round_snapshot = self.session.round_snapshots[-1]
        else:
            round_snapshot = next(
                (rs for rs in self.session.round_snapshots if rs.round_num == round_num),
                None
            )
            if not round_snapshot:
                raise ValueError(f"Round {round_num} not found")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        belief_errors = []
        profits = []
        rhos = []
        
        for agent_snap in round_snapshot.agent_snapshots:
            belief_error = abs(agent_snap.belief - round_snapshot.ground_truth)
            belief_errors.append(belief_error)
            profits.append(agent_snap.total_pnl)
            rhos.append(agent_snap.rho)
        
        # Scatter with rho-based coloring
        scatter = ax.scatter(
            belief_errors, profits, c=rhos, s=100, alpha=0.6, cmap='viridis', edgecolors='black'
        )
        
        # Add colorbar for rho
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Risk Aversion (ρ)", fontsize=11)
        
        ax.set_xlabel("Belief Error |p - ground_truth|", fontsize=12)
        ax.set_ylabel("Total Profit ($)", fontsize=12)
        ax.set_title(
            f"Belief Accuracy vs Profit (Round {round_snapshot.round_num})",
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label="Break-even")
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_rho_profit_distribution(
        self,
        round_num: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Figure:
        """
        Box plot and violin plot of profit distribution by risk aversion (rho).
        
        Args:
            round_num: if specified, show only this round; else use final round
            output_path: if provided, save figure to this path
            
        Returns:
            matplotlib Figure
        """
        if not self.session.round_snapshots:
            raise ValueError("No data to plot")
        
        if round_num is None:
            round_snapshot = self.session.round_snapshots[-1]
        else:
            round_snapshot = next(
                (rs for rs in self.session.round_snapshots if rs.round_num == round_num),
                None
            )
            if not round_snapshot:
                raise ValueError(f"Round {round_num} not found")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Group agents by rho
        rho_groups: Dict[float, List[float]] = {}
        for agent_snap in round_snapshot.agent_snapshots:
            rho = agent_snap.rho
            if rho not in rho_groups:
                rho_groups[rho] = []
            rho_groups[rho].append(agent_snap.total_pnl)
        
        rhos = sorted(rho_groups.keys())
        profits_by_rho = [rho_groups[rho] for rho in rhos]
        rho_labels = [f"{rho:.2f}" for rho in rhos]
        
        # Box plot
        bp = ax1.boxplot(profits_by_rho, labels=rho_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(rhos))):
            patch.set_facecolor(color)
        ax1.set_xlabel("Risk Aversion (ρ)", fontsize=12)
        ax1.set_ylabel("Profit ($)", fontsize=12)
        ax1.set_title("Profit Distribution by Risk Aversion (Box Plot)", fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Violin plot
        data_for_violin = []
        rho_labels_expanded = []
        for rho, profits in zip(rhos, profits_by_rho):
            for profit in profits:
                data_for_violin.append(profit)
                rho_labels_expanded.append(f"{rho:.2f}")
        
        parts = ax2.violinplot(
            profits_by_rho,
            positions=range(len(rhos)),
            widths=0.7,
            showmeans=True,
            showmedians=True
        )
        ax2.set_xticks(range(len(rhos)))
        ax2.set_xticklabels(rho_labels)
        ax2.set_xlabel("Risk Aversion (ρ)", fontsize=12)
        ax2.set_ylabel("Profit ($)", fontsize=12)
        ax2.set_title("Profit Distribution by Risk Aversion (Violin Plot)", fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_inequality_metrics(
        self,
        output_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot inequality metrics over time: Gini, std dev, range.
        
        Args:
            output_path: if provided, save figure to this path
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        rounds = [rs.round_num for rs in self.session.round_snapshots]
        ginis = [rs.gini_coefficient for rs in self.session.round_snapshots]
        std_devs = [rs.std_profit for rs in self.session.round_snapshots]
        ranges = [rs.max_profit - rs.min_profit for rs in self.session.round_snapshots]
        max_profits = [rs.max_profit for rs in self.session.round_snapshots]
        min_profits = [rs.min_profit for rs in self.session.round_snapshots]
        
        # Gini coefficient
        axes[0, 0].plot(rounds, ginis, linewidth=2.5, marker='o', color='steelblue')
        axes[0, 0].set_ylabel("Gini Coefficient", fontsize=11)
        axes[0, 0].set_title("Income Inequality (Gini) Over Time", fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Std Dev
        axes[0, 1].plot(rounds, std_devs, linewidth=2.5, marker='s', color='coral')
        axes[0, 1].set_ylabel("Std Dev of Profit ($)", fontsize=11)
        axes[0, 1].set_title("Profit Volatility Over Time", fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Profit range
        axes[1, 0].fill_between(rounds, min_profits, max_profits, alpha=0.3, color='green')
        axes[1, 0].plot(rounds, max_profits, label="Max Profit", linewidth=2, marker='^', color='darkgreen')
        axes[1, 0].plot(rounds, min_profits, label="Min Profit", linewidth=2, marker='v', color='darkred')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 0].set_ylabel("Profit ($)", fontsize=11)
        axes[1, 0].set_title("Profit Range (Max - Min)", fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Profit range magnitude
        axes[1, 1].plot(ranges, linewidth=2.5, marker='D', color='purple')
        axes[1, 1].set_ylabel("Profit Range ($)", fontsize=11)
        axes[1, 1].set_title("Magnitude of Profit Spread", fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel("Round", fontsize=11)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cumulative_volume_and_price(
        self,
        output_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot cumulative trading volume and market price evolution.
        
        Args:
            output_path: if provided, save figure to this path
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        rounds = [rs.round_num for rs in self.session.round_snapshots]
        prices = [rs.market_price for rs in self.session.round_snapshots]
        volumes = [rs.total_volume for rs in self.session.round_snapshots]
        ground_truths = [rs.ground_truth for rs in self.session.round_snapshots]
        
        cumulative_volume = np.cumsum(volumes)
        
        # Price evolution
        ax1.plot(rounds, prices, label="Market Price", linewidth=2.5, marker='o', color='steelblue')
        ax1.plot(rounds, ground_truths, label="Ground Truth", linewidth=2.5, 
                 linestyle='--', marker='s', color='darkred')
        ax1.set_ylabel("Price", fontsize=12)
        ax1.set_title("Market Price vs Ground Truth", fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Cumulative volume
        ax2.bar(rounds, volumes, alpha=0.6, color='coral', label='Volume per Round')
        ax2.plot(rounds, cumulative_volume, color='darkgreen', linewidth=2.5, 
                 marker='o', label='Cumulative Volume')
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Share Volume", fontsize=12)
        ax2.set_title("Trading Volume", fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_final_profit_distribution(
        self,
        output_path: Optional[str] = None,
    ) -> Figure:
        """
        Histogram of final profit distribution across all agents.
        
        Args:
            output_path: if provided, save figure to this path
            
        Returns:
            matplotlib Figure
        """
        if not self.session.round_snapshots:
            raise ValueError("No data to plot")
        
        final_round = self.session.round_snapshots[-1]
        profits = [s.total_pnl for s in final_round.agent_snapshots]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(profits, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(profits), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(profits):.2f}')
        ax.axvline(np.median(profits), color='green', linestyle='--', linewidth=2, label=f'Median: ${np.median(profits):.2f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel("Profit ($)", fontsize=12)
        ax.set_ylabel("Number of Agents", fontsize=12)
        ax.set_title("Final Profit Distribution", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(
        self,
        output_dir: str = "outputs",
        run_name: str = "profitability_run",
    ) -> Dict[str, str]:
        """
        Generate all profitability visualizations.
        
        Args:
            output_dir: directory to save plots
            run_name: prefix for output file names
            
        Returns:
            Mapping from plot name to file path
        """
        from pathlib import Path
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        try:
            # Profit timeseries
            fig = self.plot_profit_timeseries(aggregate_only=True)
            path = out_path / f"{run_name}_profit_timeseries.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plots["profit_timeseries"] = str(path)
            plt.close(fig)
            
            # Individual agent timeseries (only if not too many agents)
            if len(self.session.agent_trackers) <= 25:
                fig = self.plot_profit_timeseries(aggregate_only=False)
                path = out_path / f"{run_name}_agent_timeseries.png"
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plots["agent_timeseries"] = str(path)
                plt.close(fig)
            
            # Belief vs profit
            fig = self.plot_belief_accuracy_vs_profit()
            path = out_path / f"{run_name}_belief_vs_profit.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plots["belief_vs_profit"] = str(path)
            plt.close(fig)
            
            # Rho distribution
            fig = self.plot_rho_profit_distribution()
            path = out_path / f"{run_name}_rho_distribution.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plots["rho_distribution"] = str(path)
            plt.close(fig)
            
            # Inequality metrics
            fig = self.plot_inequality_metrics()
            path = out_path / f"{run_name}_inequality.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plots["inequality"] = str(path)
            plt.close(fig)
            
            # Volume and price
            fig = self.plot_cumulative_volume_and_price()
            path = out_path / f"{run_name}_volume_price.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plots["volume_price"] = str(path)
            plt.close(fig)
            
            # Final distribution
            fig = self.plot_final_profit_distribution()
            path = out_path / f"{run_name}_final_distribution.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plots["final_distribution"] = str(path)
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Error generating plots: {e}")
        
        return plots
