"""Paired, reproducible strategy evaluation and reports."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from nfl_coaching_sim.models import BenchmarkResult, DecisionTrace, Scenario
from nfl_coaching_sim.simulator import DeterministicSimulator


class CoachingStrategy(Protocol):
    name: str

    def decide(self, scenario: Scenario) -> DecisionTrace: ...


def run_benchmark(
    scenarios: Sequence[Scenario],
    strategies: Sequence[CoachingStrategy],
    simulator: DeterministicSimulator,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for scenario in scenarios:
        candidates = simulator.candidates(scenario)
        oracle = max(value[0] for value in candidates.values())
        for strategy in strategies:
            trace = strategy.decide(scenario)
            value = simulator.score(scenario, trace.decision)
            results.append(
                BenchmarkResult(
                    scenario_id=scenario.scenario_id,
                    strategy=strategy.name,
                    decision=trace.decision,
                    expected_wpa=value.expected_wpa,
                    expected_epa=value.expected_epa,
                    oracle_regret=value.oracle_regret,
                    best_action=abs(value.expected_wpa - oracle) < 1e-12,
                    latency_seconds=trace.latency_seconds,
                    model_calls=trace.model_calls,
                    fallback_used=trace.fallback_used,
                    failures=trace.failures,
                    model_id=trace.model_id,
                )
            )
    return results


def paired_bootstrap(values: Sequence[float], seed: int = 2026, samples: int = 2000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate(results: Iterable[BenchmarkResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[result.strategy].append(result)
    summary: dict[str, dict[str, float]] = {}
    for strategy, rows in sorted(grouped.items()):
        wpa = [row.expected_wpa for row in rows]
        low, high = paired_bootstrap(wpa)
        summary[strategy] = {
            "scenarios": float(len(rows)),
            "mean_expected_wpa": float(np.mean(wpa)),
            "wpa_ci_low": low,
            "wpa_ci_high": high,
            "mean_expected_epa": float(np.mean([row.expected_epa for row in rows])),
            "mean_oracle_regret": float(np.mean([row.oracle_regret for row in rows])),
            "best_action_rate": float(np.mean([row.best_action for row in rows])),
            "fallback_rate": float(np.mean([row.fallback_used for row in rows])),
            "failure_rate": float(np.mean([bool(row.failures) for row in rows])),
            "mean_latency_seconds": float(np.mean([row.latency_seconds for row in rows])),
            "mean_model_calls": float(np.mean([row.model_calls for row in rows])),
        }
    return summary


def write_results(results: Sequence[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(row.model_dump_json() for row in results) + "\n",
        encoding="utf-8",
    )


def read_results(path: Path) -> list[BenchmarkResult]:
    return [BenchmarkResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_report(results: Sequence[BenchmarkResult], path: Path) -> None:
    summary = aggregate(results)
    headings = (
        "Decision-Maker",
        "Game Situations",
        "Average WPA",
        "95% CI",
        "Average EPA",
        "Regret vs. Best Call",
        "Won the Call",
        "Backup Calls",
        "Headset Errors",
        "Time to Send In Call",
        "Model Calls",
    )
    rows = []
    for name, values in summary.items():
        cells = (
            name,
            f"{int(values['scenarios'])}",
            f"{values['mean_expected_wpa']:.4f}",
            f"[{values['wpa_ci_low']:.4f}, {values['wpa_ci_high']:.4f}]",
            f"{values['mean_expected_epa']:.4f}",
            f"{values['mean_oracle_regret']:.4f}",
            f"{values['best_action_rate']:.1%}",
            f"{values['fallback_rate']:.1%}",
            f"{values['failure_rate']:.1%}",
            f"{values['mean_latency_seconds']:.2f}s",
            f"{values['mean_model_calls']:.1f}",
        )
        rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>NFL Postgame Decision Report</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;border:1px solid #ccc;text-align:right}}
th:first-child,td:first-child{{text-align:left}}caption{{text-align:left;font-size:1.5rem;font-weight:700;margin-bottom:1rem}}</style>
</head><body><table><caption>NFL Coaching Staff — Postgame Decision Report</caption>
<thead><tr>{''.join(f'<th>{html.escape(item)}</th>' for item in headings)}</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>Primary grade: expected win-probability added. The regret column measures how much value the call left on the field
versus the simulator's best available call. Confidence intervals use a fixed-seed paired bootstrap.
This is an observational decision benchmark, not a claim that a different historical play call would have caused the modeled outcome.</p>
<script type="application/json" id="summary">{html.escape(json.dumps(summary, sort_keys=True))}</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
