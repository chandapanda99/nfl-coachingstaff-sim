import asyncio
import json

import httpx
from typer.testing import CliRunner

from nfl_coaching_sim.api import create_api
from nfl_coaching_sim.cli import app
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.runtime import migrate_legacy_user_data, user_data_root
from nfl_coaching_sim.settings import ApplicationSettings
from nfl_coaching_sim.simulator import DeterministicSimulator


def test_local_api_covers_scenarios_persistence_settings_and_streaming(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    legacy_data = tmp_path / "local-app-data" / "com.aayushchanda.nfl-coachingstaff-sim"
    legacy_logs = tmp_path / "local-app-data" / "NFL Virtual Coaching Staff"
    (legacy_data / "config").mkdir(parents=True)
    (legacy_data / "data").mkdir()
    (legacy_data / "logs").mkdir()
    (legacy_data / "EBWebView").mkdir()
    legacy_logs.mkdir(parents=True)
    (legacy_data / "config" / ".env").write_text("NFL_COACH_LOG_LEVEL=INFO\n", encoding="utf-8")
    (legacy_data / "data" / "custom-scenarios.jsonl").write_text("", encoding="utf-8")
    (legacy_data / "logs" / "sidecar.log").write_text("packaged launch\n", encoding="utf-8")
    (legacy_logs / "sidecar.log").write_text("previous launch\n", encoding="utf-8")
    migrate_legacy_user_data()
    canonical_data = user_data_root()
    assert canonical_data.name == "nfl-coachingstaff-sim"
    assert (canonical_data / "config" / ".env").is_file()
    assert (canonical_data / "data" / "custom-scenarios.jsonl").is_file()
    assert (canonical_data / "logs" / "sidecar.log").is_file()
    assert (canonical_data / "EBWebView").is_dir()
    assert not legacy_data.exists()
    assert not legacy_logs.exists()

    cli_help = CliRunner().invoke(app, ["--help"])
    assert cli_help.exit_code == 0
    assert "serve" in cli_help.stdout

    custom_path = tmp_path / "custom-scenarios.jsonl"
    defaults = ApplicationSettings(
        provider="azure_foundry",
        model="foundry-deployment",
        base_url="https://example.services.ai.azure.com/openai/v1/",
        upstream_url="https://example.org/open-model",
        model_license="Apache-2.0",
        reasoning_effort="medium",
        foundry_api_key="secret-not-for-the-client",
    )
    scenarios = demo_scenarios()
    api = create_api(
        scenarios,
        DeterministicSimulator(),
        custom_scenarios_path=custom_path,
        settings=defaults,
    )

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/health")).json() == {"status": "ready"}
            settings = (await client.get("/api/settings")).json()
            assert settings["provider"] == "azure_foundry"
            assert settings["api_key_configured"] is True
            assert "secret-not-for-the-client" not in json.dumps(settings)

            initial = (await client.get("/api/scenarios")).json()
            assert len(initial) == len(scenarios)
            assert all(item["library"] == "prebuilt" for item in initial)

            custom_values = {
                "name": "Must-Have Fourth Down",
                "season": 2025,
                "week": 18,
                "possession_team": "CHI",
                "defensive_team": "GB",
                "possession_score": 24,
                "defensive_score": 27,
                "quarter": 4,
                "clock": "1:12",
                "down": 4,
                "yards_to_go": 3,
                "field_side": "defense",
                "yard_line": 38,
                "possession_timeouts": 1,
                "defensive_timeouts": 2,
                "win_probability_percent": 17.5,
                "expected_points": 1.25,
            }
            assert (
                await client.put(f"/api/scenarios/{scenarios[0].scenario_id}", json=custom_values)
            ).status_code == 404
            created = await client.post("/api/scenarios", json=custom_values)
            assert created.status_code == 201
            custom = created.json()
            custom_id = custom["scenario"]["scenario_id"]
            assert custom["library"] == "custom"
            assert custom["scenario"]["state"]["yardline_100"] == 38

            custom_values["name"] = "Edited Fourth Down"
            updated = await client.put(f"/api/scenarios/{custom_id}", json=custom_values)
            assert updated.status_code == 200
            updated_id = updated.json()["scenario"]["scenario_id"]
            assert updated.json()["scenario"]["name"] == "Edited Fourth Down"
            assert len((await client.get("/api/scenarios?library=custom")).json()) == 1

            async with client.stream(
                "POST",
                "/api/deliberations/stream",
                json={"scenario_id": scenarios[0].scenario_id, "strategy": "expected_points"},
            ) as response:
                events = [json.loads(line) async for line in response.aiter_lines() if line]
            assert response.status_code == 200
            assert events[0]["stage"] == "started"
            assert events[-1]["stage"] == "completed"
            assert events[-1]["trace"]["strategy"] == "expected_points"
            assert events[-1]["score"]["simulator_version"] == "1.0"

            assert (await client.delete(f"/api/scenarios/{updated_id}")).status_code == 204
            assert (await client.get("/api/scenarios?library=custom")).json() == []

    asyncio.run(exercise_api())
