from nfl_coaching_sim.app import create_app, run_strategy_events, scenario_view
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.simulator import DeterministicSimulator


def test_gradio_app_builds_and_critical_callbacks_run_without_model_server() -> None:
    scenarios = demo_scenarios()
    payloads = [item.model_dump(mode="json") for item in scenarios]
    scoreboard, state, baseline = scenario_view(scenarios[0].scenario_id, payloads)
    events = list(
        run_strategy_events(
            scenarios[0].scenario_id,
            "Expected points",
            "ollama",
            "",
            "",
            "",
            "",
            payloads,
            DeterministicSimulator(),
        )
    )
    app = create_app(scenarios[:2], DeterministicSimulator())

    assert "Q" in scoreboard
    assert state["down"] == scenarios[0].state.down
    assert baseline
    assert events[-1][2]["action"]
    assert app is not None
