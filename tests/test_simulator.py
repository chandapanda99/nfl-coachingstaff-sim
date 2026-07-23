from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.models import Action, Decision
from nfl_coaching_sim.simulator import DeterministicSimulator


def _training_row(index: int) -> dict:
    scenario = demo_scenarios()[index % 25]
    state = scenario.state
    play_type = "run" if index % 2 == 0 else "pass"
    return {
        "game_id": state.game_id,
        "play_id": index,
        "season": 2023,
        "week": state.week,
        "qtr": state.quarter,
        "game_seconds_remaining": state.game_seconds_remaining,
        "down": min(state.down, 3),
        "ydstogo": state.yards_to_go,
        "yardline_100": state.yardline_100,
        "posteam": state.possession_team,
        "defteam": state.defensive_team,
        "posteam_score": state.possession_score,
        "defteam_score": state.defensive_score,
        "posteam_timeouts_remaining": state.possession_timeouts,
        "defteam_timeouts_remaining": state.defensive_timeouts,
        "wp": state.win_probability,
        "ep": state.expected_points,
        "epa": 0.2 if play_type == "pass" else -0.05,
        "wpa": 0.015 if play_type == "pass" else -0.004,
        "play_type": play_type,
        "no_play": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
    }


def test_simulator_is_deterministic_reloadable_and_has_offline_fallback(tmp_path) -> None:
    scenario = demo_scenarios()[1]
    decision = Decision(action=Action.PASS, rationale="test")
    first_fallback = DeterministicSimulator().score(scenario, decision)
    second_fallback = DeterministicSimulator().score(scenario, decision)
    assert first_fallback == second_fallback

    simulator = DeterministicSimulator().fit(_training_row(i) for i in range(60))
    trained_value = simulator.score(scenario, decision)
    artifact = tmp_path / "simulator.joblib"
    simulator.save(artifact)
    loaded_value = DeterministicSimulator.load(artifact).score(scenario, decision)

    assert trained_value == loaded_value
    assert loaded_value.oracle_regret >= 0
    assert set(simulator.candidates(scenario)) == set(scenario.state.legal_actions)
