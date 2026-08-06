"""Local HTTP API used by the web client and Tauri desktop shell."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.models import Scenario
from nfl_coaching_sim.services import CoachingApplication, CustomScenarioInput, DeliberationInput, ScenarioRepository
from nfl_coaching_sim.settings import ApplicationSettings, get_application_settings
from nfl_coaching_sim.simulator import DeterministicSimulator


class ScenarioEnvelope(BaseModel):
    scenario: Scenario
    display_name: str
    library: Literal["prebuilt", "custom"]


class SettingsView(BaseModel):
    provider: str
    model: str
    base_url: str
    upstream_url: str
    model_license: str
    reasoning_effort: str | None
    api_key_configured: bool


def _envelope(scenario: Scenario) -> ScenarioEnvelope:
    return ScenarioEnvelope(
        scenario=scenario,
        display_name=scenario.display_name,
        library="custom" if scenario.scenario_id.startswith("custom-") else "prebuilt",
    )


def create_api(
    scenarios: Sequence[Scenario] | None = None,
    simulator: DeterministicSimulator | None = None,
    custom_scenarios_path: Path | None = None,
    settings: ApplicationSettings | None = None,
    frontend_dist: Path | None = None,
) -> FastAPI:
    resolved_settings = settings or get_application_settings()
    service = CoachingApplication(
        ScenarioRepository(scenarios or demo_scenarios(), custom_scenarios_path),
        simulator or DeterministicSimulator(),
        resolved_settings,
    )
    api = FastAPI(title="NFL Virtual Coaching Staff API", version="1.0.0")
    api.state.coaching = service
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "tauri://localhost",
            "https://tauri.localhost",
        ],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @api.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @api.get("/api/settings", response_model=SettingsView)
    def application_settings() -> SettingsView:
        current = service.settings
        return SettingsView(
            provider=current.provider,
            model=current.model,
            base_url=current.base_url,
            upstream_url=current.upstream_url,
            model_license=current.model_license,
            reasoning_effort=current.reasoning_effort,
            api_key_configured=bool(current.foundry_api_key),
        )

    @api.get("/api/scenarios", response_model=list[ScenarioEnvelope])
    def list_scenarios(library: Literal["prebuilt", "custom", "all"] = Query(default="all")) -> list[ScenarioEnvelope]:
        return [_envelope(item) for item in service.scenarios.list(library)]

    @api.get("/api/scenarios/{scenario_id}", response_model=ScenarioEnvelope)
    def get_scenario(scenario_id: str) -> ScenarioEnvelope:
        try:
            return _envelope(service.scenarios.get(scenario_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @api.post("/api/scenarios", response_model=ScenarioEnvelope, status_code=201)
    def create_scenario(values: CustomScenarioInput) -> ScenarioEnvelope:
        try:
            return _envelope(service.scenarios.save(values))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @api.put("/api/scenarios/{scenario_id}", response_model=ScenarioEnvelope)
    def update_scenario(scenario_id: str, values: CustomScenarioInput) -> ScenarioEnvelope:
        try:
            return _envelope(service.scenarios.save(values, replacing_scenario_id=scenario_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @api.delete("/api/scenarios/{scenario_id}", status_code=204)
    def remove_scenario(scenario_id: str) -> None:
        try:
            service.scenarios.delete(scenario_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @api.post("/api/deliberations/stream")
    def stream_deliberation(request: DeliberationInput) -> StreamingResponse:
        def stream():
            try:
                for event in service.iter_deliberation(request):
                    yield event.model_dump_json() + "\n"
            except KeyError as error:
                yield json.dumps({"stage": "error", "message": str(error)}) + "\n"
            except Exception as error:
                yield json.dumps({"stage": "error", "message": f"Headset communication failed: {error}"}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    if frontend_dist is not None and (frontend_dist / "index.html").exists():
        api.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return api
