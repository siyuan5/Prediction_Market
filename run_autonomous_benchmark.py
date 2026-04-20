# autonomous mode evaluation benchmarks
# runs 4 experiments, saves csv/json/png to outputs/autonomous/
#
# experiments:
#   1. autonomous vs round-based convergence
#   2. personality diversity effect
#   3. news event response and persistence
#   4. multi-market validation
#
# how to run: python run_autonomous_benchmark.py [--experiment 1|2|3|4|all]

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import uvicorn

# let us import from src/
_REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

OUTPUT_DIR = _REPO / "outputs" / "autonomous"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# uvicorn in background thread
# ----------------------------------------------------------------------

class _BackgroundServer:
    # runs the fastapi app in a thread so agents can hit real http endpoints

    def __init__(self, port: int, db_path: str):
        self.port = port
        self.db_path = db_path
        self.base_url = f"http://127.0.0.1:{port}"
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        os.environ["MARKET_DB_PATH"] = self.db_path
        os.environ["AUTONOMOUS_API_BASE"] = f"{self.base_url}/api"

        # reset singletons before importing, so they pick up the fresh db path + url
        from api.market_routes import reset_market_runtime
        reset_market_runtime()

        from api.main import app

        cfg = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # wait for it to be ready
        for _ in range(50):
            try:
                r = requests.get(f"{self.base_url}/docs", timeout=0.5)
                if r.status_code < 500:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
        raise RuntimeError("server did not come up in time")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        from api.market_routes import reset_market_runtime
        reset_market_runtime()


_port_counter = [8100]


def _next_port() -> int:
    # each server gets a unique port to avoid collisions within a single run
    _port_counter[0] += 1
    return _port_counter[0]


@contextmanager
def running_server():
    # context manager so experiments are isolated from each other
    tmp = tempfile.mkdtemp(prefix="bench_")
    db_path = os.path.join(tmp, "bench.sqlite")
    port = _next_port()
    server = _BackgroundServer(port=port, db_path=db_path)
    try:
        server.start()
        yield server
    finally:
        server.stop()
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# http client wrappers
# ----------------------------------------------------------------------

def api_create_market(base_url: str, *, mechanism: str, ground_truth: float, b: float = 100.0,
                       title: str = "benchmark market") -> Dict[str, Any]:
    r = requests.post(f"{base_url}/api/market/create", json={
        "mechanism": mechanism,
        "ground_truth": ground_truth,
        "b": b,
        "title": title,
    })
    r.raise_for_status()
    return r.json()


def api_create_agent(base_url: str, *, name: str, cash: float, belief: float, rho: float,
                      personality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {"name": name, "cash": cash, "belief": belief, "rho": rho}
    if personality is not None:
        payload["personality"] = personality
    r = requests.post(f"{base_url}/api/agents", json=payload)
    r.raise_for_status()
    return r.json()


def api_join(base_url: str, market_id: int, agent_id: int):
    r = requests.post(f"{base_url}/api/market/{market_id}/join", json={"agent_id": agent_id})
    r.raise_for_status()
    return r.json()


def api_start(base_url: str, market_id: int):
    r = requests.post(f"{base_url}/api/market/{market_id}/start")
    r.raise_for_status()
    return r.json()


def api_stop(base_url: str, market_id: int):
    r = requests.post(f"{base_url}/api/market/{market_id}/stop")
    r.raise_for_status()
    return r.json()


def api_detail(base_url: str, market_id: int):
    r = requests.get(f"{base_url}/api/market/{market_id}/detail")
    r.raise_for_status()
    return r.json()


def api_price(base_url: str, market_id: int) -> float:
    r = requests.get(f"{base_url}/api/market/{market_id}/price")
    r.raise_for_status()
    return float(r.json()["price"])


def api_news(base_url: str, market_id: int, *, headline: str, new_belief: float,
              affected_fraction: float = 0.5, min_signal_sensitivity: float = 0.3):
    r = requests.post(f"{base_url}/api/market/{market_id}/news", json={
        "headline": headline,
        "new_belief": new_belief,
        "affected_fraction": affected_fraction,
        "min_signal_sensitivity": min_signal_sensitivity,
    })
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------

def sample_gaussian_beliefs(seed: int, ground_truth: float, sigma: float, n: int) -> List[float]:
    rng = np.random.default_rng(seed)
    vals = rng.normal(loc=ground_truth, scale=sigma, size=n)
    return list(np.clip(vals, 0.01, 0.99))


def sample_rhos(seed: int, n: int) -> List[float]:
    rng = np.random.default_rng(seed + 1000)
    choices = [0.5, 1.0, 2.0]
    return list(rng.choice(choices, size=n))


def _default_personality(i: int, seed: int) -> Dict[str, Any]:
    # higher edge threshold + lower participation rate damps oscillation
    # (autonomous agents don't have trade_fraction the way round-based does)
    rng = np.random.default_rng(seed * 100 + i)
    return {
        "check_interval_mean": float(rng.uniform(2.0, 4.0)),
        "check_interval_jitter": 1.0,
        "edge_threshold": float(rng.uniform(0.05, 0.12)),
        "participation_rate": float(rng.uniform(0.4, 0.7)),
        "trade_size_noise": 0.15,
        "signal_sensitivity": 0.5,
        "stubbornness": 0.3,
    }


def setup_autonomous_market(server: _BackgroundServer, *, mechanism: str, ground_truth: float,
                              n_agents: int, seed: int, b: float = 100.0,
                              personality_fn=None, title: str = "benchmark") -> Dict[str, Any]:
    # creates a market + agents, joins them, returns market_id and agent_ids
    base = server.base_url
    mkt = api_create_market(base, mechanism=mechanism, ground_truth=ground_truth, b=b, title=title)
    market_id = mkt["market_id"]

    beliefs = sample_gaussian_beliefs(seed, ground_truth, 0.10, n_agents)
    rhos = sample_rhos(seed, n_agents)

    pfn = personality_fn or _default_personality

    agent_ids = []
    for i in range(n_agents):
        personality = pfn(i, seed)
        agent = api_create_agent(
            base, name=f"{title}_agent_{seed}_{i}",
            cash=100.0, belief=float(beliefs[i]), rho=float(rhos[i]),
            personality=personality,
        )
        agent_ids.append(agent["agent_id"])
        api_join(base, market_id, agent["agent_id"])

    return {"market_id": market_id, "agent_ids": agent_ids}


def run_autonomous(server: _BackgroundServer, market_id: int, duration_sec: float,
                    price_sample_hz: float = 2.0) -> Dict[str, Any]:
    # starts autonomous trading, samples price at given rate, stops, returns trajectory
    base = server.base_url
    api_start(base, market_id)

    start_time = time.time()
    times = []
    prices = []

    sample_period = 1.0 / price_sample_hz
    while time.time() - start_time < duration_sec:
        elapsed = time.time() - start_time
        try:
            p = api_price(base, market_id)
            times.append(elapsed)
            prices.append(p)
        except requests.RequestException:
            pass
        time.sleep(sample_period)

    stop_info = api_stop(base, market_id)
    final_detail = api_detail(base, market_id)

    return {
        "times": times,
        "prices": prices,
        "final_price": float(final_detail["price"]),
        "trade_count": int(stop_info.get("total_trades", 0)),
        "duration_sec": float(stop_info.get("duration_sec", duration_sec)),
    }


def run_round_based(*, seed: int, ground_truth: float, n_agents: int, n_rounds: int,
                     mechanism: str = "lmsr", b: float = 100.0) -> Dict[str, Any]:
    # runs the old SimulationEngine for comparison
    from simulation_engine import SimulationEngine
    from belief_init import BeliefSpec

    engine = SimulationEngine(
        mechanism=mechanism, phase=2, seed=seed, ground_truth=ground_truth,
        n_agents=n_agents, initial_cash=100.0, b=b,
        belief_spec=BeliefSpec(mode="gaussian", sigma=0.10),
    )
    engine.run(n_rounds)
    metrics = engine.get_metrics()

    return {
        "price_series": metrics["price_series"],
        "error_series": metrics["error_series"],
        "final_price": metrics["final_price"],
        "final_error": metrics["final_error"],
    }


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    return {"mean": m, "std": s}


# ----------------------------------------------------------------------
# experiment 1: autonomous vs round-based convergence
# ----------------------------------------------------------------------

def experiment_1_convergence(seeds=(42, 101, 777), duration_sec: float = 60.0,
                              n_rounds: int = 100, n_agents: int = 15) -> Dict[str, Any]:
    ground_truth = 0.70

    auto_prices = []
    auto_errors = []
    round_prices = []
    round_errors = []

    print(f"\n[exp1] autonomous vs round-based (seeds={seeds})")

    for seed in seeds:
        print(f"  seed {seed}...")
        with running_server() as server:
            setup = setup_autonomous_market(
                server, mechanism="lmsr", ground_truth=ground_truth,
                n_agents=n_agents, seed=seed, title="exp1",
            )
            result = run_autonomous(server, setup["market_id"], duration_sec=duration_sec)
            auto_prices.append(result["final_price"])
            auto_errors.append(abs(result["final_price"] - ground_truth))

        rb = run_round_based(seed=seed, ground_truth=ground_truth,
                              n_agents=n_agents, n_rounds=n_rounds)
        round_prices.append(rb["final_price"])
        round_errors.append(rb["final_error"])

    rows = []
    for i, seed in enumerate(seeds):
        rows.append({
            "seed": seed,
            "mode": "autonomous",
            "final_price": auto_prices[i],
            "final_error": auto_errors[i],
        })
        rows.append({
            "seed": seed,
            "mode": "round_based",
            "final_price": round_prices[i],
            "final_error": round_errors[i],
        })
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "exp1_convergence.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "autonomous": {
            "final_price": mean_std(auto_prices),
            "final_error": mean_std(auto_errors),
        },
        "round_based": {
            "final_price": mean_std(round_prices),
            "final_error": mean_std(round_errors),
        },
        "ground_truth": ground_truth,
        "n_seeds": len(seeds),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, auto_errors, w, label="autonomous", alpha=0.8)
    ax.bar(x + w/2, round_errors, w, label="round-based", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("|final price - ground truth|")
    ax.set_title("exp 1: convergence error (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp1_convergence.png", dpi=120)
    plt.close(fig)

    print(f"  autonomous error: {summary['autonomous']['final_error']['mean']:.4f} "
          f"+/- {summary['autonomous']['final_error']['std']:.4f}")
    print(f"  round-based error: {summary['round_based']['final_error']['mean']:.4f} "
          f"+/- {summary['round_based']['final_error']['std']:.4f}")

    return summary


# ----------------------------------------------------------------------
# experiment 2: personality diversity effect
# ----------------------------------------------------------------------

def _personality_identical(i: int, seed: int) -> Dict[str, Any]:
    return {
        "check_interval_mean": 2.0, "check_interval_jitter": 0.0,
        "edge_threshold": 0.03, "participation_rate": 0.8,
        "trade_size_noise": 0.0, "signal_sensitivity": 0.5, "stubbornness": 0.3,
    }


def _personality_moderate(i: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed * 100 + i)
    return {
        "check_interval_mean": float(rng.uniform(1.5, 3.0)),
        "check_interval_jitter": 0.5,
        "edge_threshold": float(rng.uniform(0.02, 0.05)),
        "participation_rate": float(rng.uniform(0.7, 0.9)),
        "trade_size_noise": float(rng.uniform(0.05, 0.15)),
        "signal_sensitivity": 0.5, "stubbornness": 0.3,
    }


def _personality_high(i: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed * 100 + i)
    return {
        "check_interval_mean": float(rng.uniform(1.0, 4.0)),
        "check_interval_jitter": float(rng.uniform(0.5, 1.5)),
        "edge_threshold": float(rng.uniform(0.01, 0.10)),
        "participation_rate": float(rng.uniform(0.4, 1.0)),
        "trade_size_noise": float(rng.uniform(0.1, 0.3)),
        "signal_sensitivity": float(rng.uniform(0.2, 0.9)),
        "stubbornness": float(rng.uniform(0.1, 0.6)),
    }


def experiment_2_personality(seeds=(42, 101, 777), duration_sec: float = 45.0,
                               n_agents: int = 15) -> Dict[str, Any]:
    ground_truth = 0.70

    configs = [
        ("identical", _personality_identical),
        ("moderate_variety", _personality_moderate),
        ("high_variety", _personality_high),
    ]

    print(f"\n[exp2] personality diversity (seeds={seeds})")

    rows = []
    summary_by_config = {}

    for config_name, pfn in configs:
        print(f"  config: {config_name}")
        errors = []
        trade_counts = []
        stds = []

        for seed in seeds:
            print(f"    seed {seed}...")
            with running_server() as server:
                setup = setup_autonomous_market(
                    server, mechanism="lmsr", ground_truth=ground_truth,
                    n_agents=n_agents, seed=seed, personality_fn=pfn,
                    title=f"exp2_{config_name}",
                )
                result = run_autonomous(server, setup["market_id"], duration_sec=duration_sec)

            final_err = abs(result["final_price"] - ground_truth)
            errors.append(final_err)
            trade_counts.append(result["trade_count"])
            # stability: std of last 20% of price samples
            tail = result["prices"][-max(5, len(result["prices"]) // 5):]
            stds.append(float(np.std(tail)) if tail else 0.0)

            rows.append({
                "seed": seed, "config": config_name,
                "final_error": final_err,
                "trade_count": result["trade_count"],
                "tail_price_std": stds[-1],
            })

        summary_by_config[config_name] = {
            "final_error": mean_std(errors),
            "trade_count": mean_std(trade_counts),
            "tail_price_std": mean_std(stds),
        }

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "exp2_personality.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    config_names = [c[0] for c in configs]
    for ax, (label, metric_key) in zip(axes, [
        ("final error", "final_error"),
        ("trade count", "trade_count"),
        ("tail price std", "tail_price_std"),
    ]):
        means = [summary_by_config[c][metric_key]["mean"] for c in config_names]
        stds = [summary_by_config[c][metric_key]["std"] for c in config_names]
        ax.bar(config_names, means, yerr=stds, alpha=0.8, capsize=5)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("exp 2: personality diversity effect")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp2_personality.png", dpi=120)
    plt.close(fig)

    for cname, s in summary_by_config.items():
        print(f"  {cname}: error={s['final_error']['mean']:.4f} "
              f"trades={s['trade_count']['mean']:.1f} "
              f"tail_std={s['tail_price_std']['mean']:.4f}")

    return {"ground_truth": ground_truth, "by_config": summary_by_config, "n_seeds": len(seeds)}


# ----------------------------------------------------------------------
# experiment 3: news event response and persistence
# ----------------------------------------------------------------------

def experiment_3_news(seeds=(42, 101, 777), baseline_sec: float = 20.0,
                       recovery_sec: float = 25.0, n_agents: int = 30) -> Dict[str, Any]:
    ground_truth = 0.50   # start from neutral, so the news shift is more visible
    news_target = 0.80
    print(f"\n[exp3] news event response (seeds={seeds})")

    rows = []
    trajectories = []

    for seed in seeds:
        print(f"  seed {seed}...")
        with running_server() as server:
            setup = setup_autonomous_market(
                server, mechanism="lmsr", ground_truth=ground_truth,
                n_agents=n_agents, seed=seed,
                personality_fn=_personality_moderate, title="exp3",
            )
            market_id = setup["market_id"]

            # baseline
            before = run_autonomous(server, market_id, duration_sec=baseline_sec)
            pre_news_price = before["final_price"]

            # trigger news
            news_event = api_news(
                server.base_url, market_id,
                headline="breaking: forecast shifts bullish",
                new_belief=news_target,
                affected_fraction=0.5, min_signal_sensitivity=0.3,
            )

            # recovery
            after = run_autonomous(server, market_id, duration_sec=recovery_sec)
            post_news_price = after["final_price"]

        distance_moved = post_news_price - pre_news_price
        target_distance = news_target - pre_news_price
        fraction_moved = distance_moved / target_distance if abs(target_distance) > 1e-6 else 0.0

        rows.append({
            "seed": seed,
            "pre_news_price": pre_news_price,
            "post_news_price": post_news_price,
            "news_target": news_target,
            "distance_moved": distance_moved,
            "fraction_toward_target": fraction_moved,
            "n_affected": news_event.get("n_affected", 0),
        })

        combined_times = [t for t in before["times"]]
        combined_prices = [p for p in before["prices"]]
        news_time_marker = combined_times[-1] if combined_times else 0.0
        offset = news_time_marker
        combined_times.extend([offset + t for t in after["times"]])
        combined_prices.extend(after["prices"])

        trajectories.append({
            "seed": seed, "times": combined_times, "prices": combined_prices,
            "news_time": news_time_marker,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "exp3_news.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for traj in trajectories:
        ax.plot(traj["times"], traj["prices"], alpha=0.7,
                label=f"seed {traj['seed']}")
        ax.axvline(traj["news_time"], color="red", linestyle="--", alpha=0.3)
    ax.axhline(ground_truth, color="gray", linestyle=":", label=f"P* = {ground_truth}")
    ax.axhline(news_target, color="green", linestyle=":", label=f"news target = {news_target}")
    ax.set_xlabel("elapsed seconds")
    ax.set_ylabel("price")
    ax.set_title("exp 3: news event response (red line = news triggered)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp3_news.png", dpi=120)
    plt.close(fig)

    mean_fraction = mean(r["fraction_toward_target"] for r in rows)
    print(f"  mean fraction moved toward news target: {mean_fraction:.3f}")

    return {
        "ground_truth": ground_truth, "news_target": news_target,
        "mean_fraction_toward_target": mean_fraction,
        "per_seed": rows,
    }


# ----------------------------------------------------------------------
# experiment 4: multi-market validation
# ----------------------------------------------------------------------

def experiment_4_multimarket(seeds=(42, 101, 777), duration_sec: float = 45.0,
                              n_agents: int = 30) -> Dict[str, Any]:
    gt_1 = 0.30
    gt_2 = 0.80

    print(f"\n[exp4] multi-market (seeds={seeds})")

    rows = []

    for seed in seeds:
        print(f"  seed {seed}...")
        with running_server() as server:
            # 2 markets
            m1 = api_create_market(server.base_url, mechanism="lmsr",
                                    ground_truth=gt_1, title="market_a")
            m2 = api_create_market(server.base_url, mechanism="lmsr",
                                    ground_truth=gt_2, title="market_b")
            market_id_1 = m1["market_id"]
            market_id_2 = m2["market_id"]

            # create agents and join both markets
            beliefs_1 = sample_gaussian_beliefs(seed, gt_1, 0.10, n_agents)
            beliefs_2 = sample_gaussian_beliefs(seed + 50, gt_2, 0.10, n_agents)
            rhos = sample_rhos(seed, n_agents)

            for i in range(n_agents):
                # each agent's initial belief is a blend; they'll pick which market based on edge
                blended = (beliefs_1[i] + beliefs_2[i]) / 2.0
                agent = api_create_agent(
                    server.base_url, name=f"exp4_{seed}_{i}",
                    cash=100.0, belief=float(blended), rho=float(rhos[i]),
                )
                api_join(server.base_url, market_id_1, agent["agent_id"])
                api_join(server.base_url, market_id_2, agent["agent_id"])

            # start both, let them run
            api_start(server.base_url, market_id_1)
            api_start(server.base_url, market_id_2)
            time.sleep(duration_sec)
            api_stop(server.base_url, market_id_1)
            api_stop(server.base_url, market_id_2)

            price_1 = api_price(server.base_url, market_id_1)
            price_2 = api_price(server.base_url, market_id_2)

        rows.append({
            "seed": seed,
            "market_a_gt": gt_1, "market_a_price": price_1,
            "market_a_error": abs(price_1 - gt_1),
            "market_b_gt": gt_2, "market_b_price": price_2,
            "market_b_error": abs(price_2 - gt_2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "exp4_multimarket.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seeds))
    w = 0.35
    prices_a = [r["market_a_price"] for r in rows]
    prices_b = [r["market_b_price"] for r in rows]
    ax.bar(x - w/2, prices_a, w, label=f"market a (P*={gt_1})", alpha=0.8)
    ax.bar(x + w/2, prices_b, w, label=f"market b (P*={gt_2})", alpha=0.8)
    ax.axhline(gt_1, color="C0", linestyle="--", alpha=0.5)
    ax.axhline(gt_2, color="C1", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("final price")
    ax.set_ylim(0, 1)
    ax.set_title("exp 4: multi-market convergence (dashes = ground truths)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp4_multimarket.png", dpi=120)
    plt.close(fig)

    err_a = mean(r["market_a_error"] for r in rows)
    err_b = mean(r["market_b_error"] for r in rows)
    print(f"  market a mean error: {err_a:.4f} (target {gt_1})")
    print(f"  market b mean error: {err_b:.4f} (target {gt_2})")

    return {
        "market_a": {"ground_truth": gt_1, "mean_error": err_a},
        "market_b": {"ground_truth": gt_2, "mean_error": err_b},
        "per_seed": rows,
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                         choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--quick", action="store_true",
                         help="shorter durations for smoke testing")
    args = parser.parse_args()

    if args.quick:
        dur1 = 15.0; dur2 = 15.0; dur3b = 8.0; dur3r = 10.0; dur4 = 15.0
        seeds = (42, 101)
    else:
        dur1 = 60.0; dur2 = 45.0; dur3b = 20.0; dur3r = 25.0; dur4 = 45.0
        seeds = (42, 101, 777)

    results = {}
    start = time.time()

    if args.experiment in ("1", "all"):
        results["exp1"] = experiment_1_convergence(seeds=seeds, duration_sec=dur1)
    if args.experiment in ("2", "all"):
        results["exp2"] = experiment_2_personality(seeds=seeds, duration_sec=dur2)
    if args.experiment in ("3", "all"):
        results["exp3"] = experiment_3_news(seeds=seeds, baseline_sec=dur3b, recovery_sec=dur3r)
    if args.experiment in ("4", "all"):
        results["exp4"] = experiment_4_multimarket(seeds=seeds, duration_sec=dur4)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    elapsed = time.time() - start
    print(f"\ndone in {elapsed:.1f}s. outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
