from nfl_coaching_sim.agents import ExpectedPointsStrategy
from nfl_coaching_sim.benchmark import aggregate, run_benchmark
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.simulator import DeterministicSimulator


def test_paired_benchmark_has_reproducible_metrics_and_provenance() -> None:
    scenarios = demo_scenarios()[:8]
    strategy = ExpectedPointsStrategy()
    first = run_benchmark(scenarios, [strategy], DeterministicSimulator())
    second = run_benchmark(scenarios, [strategy], DeterministicSimulator())

    first_summary = aggregate(first)
    second_summary = aggregate(second)
    stable_metrics = {
        key: value
        for key, value in first_summary["expected_points"].items()
        if key != "mean_latency_seconds"
    }
    stable_metrics_again = {
        key: value
        for key, value in second_summary["expected_points"].items()
        if key != "mean_latency_seconds"
    }
    assert stable_metrics == stable_metrics_again
    assert len(first) == len(scenarios)
    assert {row.scenario_id for row in first} == {item.scenario_id for item in scenarios}
    assert all(row.simulator_version and row.prompt_version for row in first)
    assert 0 <= first_summary["expected_points"]["best_action_rate"] <= 1
