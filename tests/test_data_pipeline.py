from nfl_coaching_sim.data import build_scenarios, quickstart_sample


def _row(season: int, play_id: int, down: int, play_type: str, epa: float) -> dict:
    return {
        "game_id": f"{season}_01_BUF_KC",
        "play_id": play_id,
        "season": season,
        "week": 1,
        "qtr": 4,
        "game_seconds_remaining": 240,
        "down": down,
        "ydstogo": 4,
        "yardline_100": 35,
        "posteam": "BUF",
        "defteam": "KC",
        "posteam_score": 24,
        "defteam_score": 27,
        "posteam_timeouts_remaining": 2,
        "defteam_timeouts_remaining": 1,
        "wp": 0.43,
        "ep": 1.2,
        "epa": epa,
        "wpa": epa / 50,
        "play_type": play_type,
        "no_play": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
    }


def test_pipeline_isolates_seasons_and_emits_outcome_free_scenarios() -> None:
    training = [
        _row(2023, 1, 1, "run", -0.1),
        _row(2023, 2, 1, "pass", 0.2),
        _row(2023, 3, 4, "punt", -0.4),
        _row(2023, 4, 4, "field_goal", 0.3),
        _row(2023, 5, 4, "pass", 0.5),
    ]
    evaluation = [_row(2024, index, index % 4 + 1, "pass", 99) for index in range(20)]

    first = build_scenarios(training, evaluation, limit=10)
    second = build_scenarios(training, evaluation, limit=10)

    assert [item.scenario_id for item in first] == [item.scenario_id for item in second]
    assert all(item.state.season == 2024 for item in first)
    assert {item.state.down for item in first} == {1, 2, 3, 4}
    assert all("epa" not in item.model_dump()["state"] for item in first)
    assert all(99 not in item.ep_baseline.values() for item in first)

    full_pack = build_scenarios(training, evaluation, limit=20)
    quickstart = quickstart_sample(full_pack, limit=12)
    assert len(quickstart) == 12
    assert sum(item.state.down == 4 for item in quickstart) >= 5
    assert all(
        item.state.quarter == 4 and item.state.game_seconds_remaining <= 300 and abs(item.state.score_differential) <= 8
        for item in quickstart
    )

    leaked = training + [_row(2024, 99, 1, "run", 0)]
    try:
        build_scenarios(leaked, evaluation)
    except ValueError as error:
        assert "evaluation seasons" in str(error)
    else:
        raise AssertionError("evaluation leakage was accepted")
