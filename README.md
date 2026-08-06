# NFL Virtual Coaching Staff

A local-first NFL decision simulator that puts five specialist coaches on the headset, lets them challenge and revise a call, and has a head coach send the final decision to a deterministic WPA/EPA simulator.

The project is Apache-2.0 and uses an open-source application stack. Derived nflverse scenario packs retain CC-BY-4.0 attribution. It distributes no model weights or raw nflverse datasets.

## Architecture

- **Svelte + TypeScript** renders the football field, situation builder, analytics booth, live group-chat transcript, and decision grade.
- **Tauri 2** packages that interface as a native Windows, macOS, or Linux application using the operating system's webview.
- **FastAPI** exposes typed scenario and deliberation endpoints and streams one completed coach response at a time as NDJSON.
- **LangChain** provides a provider-neutral model boundary. Azure AI Foundry is the default provider; Ollama and registered custom adapters use the same coaching workflow.
- **Typer** retains the data, training, benchmarking, reporting, and server commands.

The presentation layer contains no coaching logic. The API and CLI share `CoachingApplication`, `ScenarioRepository`, provider adapters, and the deterministic simulator.

## Quick start for development

Requirements: Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), and Node.js 20 or newer.

```powershell
Copy-Item .env.example .env
uv sync --extra test
uv run nfl-coach serve
```

In a second terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The API listens on `http://127.0.0.1:8765`; interactive API documentation is available at `/docs`.

The analytics-only strategy works without a model connection. LLM strategies display a clear headset error when the configured provider is unavailable.

## Model configuration

`.env` is the single source of startup defaults for both the API-backed interface and CLI. The checked-in `config/models.example.json` documents additional provider registrations; a local `config/models.json` can extend them without becoming a second source for defaults.

Azure AI Foundry is selected by default:

```dotenv
NFL_COACH_MODEL_PROVIDER=azure_foundry
FOUNDRY_MODEL=your-deployment-name
FOUNDRY_ENDPOINT=https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/
FOUNDRY_UPSTREAM_URL=https://project-homepage-for-provenance.example
FOUNDRY_MODEL_LICENSE=Apache-2.0
AZURE_FOUNDRY_API_KEY=
AZURE_FOUNDRY_REASONING_EFFORT=medium
```

`FOUNDRY_UPSTREAM_URL` is optional provenance metadata; it is never used to query the model. Foundry calls use LangChain's Responses API path (`use_responses_api=True`), omit temperature, and send only parameters supported by that provider. Authentication uses the API key when supplied and otherwise uses `DefaultAzureCredential`.

For Ollama:

```dotenv
NFL_COACH_MODEL_PROVIDER=ollama
OLLAMA_MODEL=your-local-model-tag
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_UPSTREAM_URL=https://model-project.example
OLLAMA_MODEL_LICENSE=Apache-2.0
```

Keys stay in the backend process and are never returned by `/api/settings` or written to benchmark artifacts.

## Desktop development and packaging

Install the stable Rust toolchain in addition to the quick-start requirements. Run the FastAPI backend in one terminal, then:

```powershell
Set-Location frontend
npm run tauri dev
```

Create a release installer on the target operating system with:

```powershell
uv sync --extra package
uv run --extra package python packaging/build_sidecar.py
Set-Location frontend
npm install
npm run tauri build
```

The sidecar script builds the Python/FastAPI backend with PyInstaller and gives it the target-triple filename Tauri expects. Tauri then produces the platform's native package. Build separately on Windows, macOS, and Linux; a package from one operating system is not portable to another.

## Research CLI

```powershell
uv run nfl-coach data sync
uv run nfl-coach scenarios build
uv run nfl-coach simulator train
uv run nfl-coach benchmark run
uv run nfl-coach benchmark report
uv run nfl-coach app serve
```

The checked-in 25-situation pack makes the app immediately usable. The 250-situation benchmark and trained simulator are reproducible from nflverse play-by-play. Training uses 2016–2023; 2024–2025 are reserved for evaluation.

## Verification

The durable Python suite intentionally contains only five critical-path tests: data isolation, simulator determinism/reload/fallback, full orchestration and provider behavior, paired benchmark reproducibility, and the application/API contract.

```powershell
uv run --extra test pytest
Set-Location frontend
npm run check
npm run build
```

## Licensing and limitations

Application code is Apache-2.0. Generated scenario packs derived from nflverse are CC-BY-4.0; see `NOTICE` and the scenario source manifests. The v1 benchmark is observational and does not eliminate historical coaching-selection bias. Overtime, penalties, kneels, spikes, injury/personnel context, and play-design granularity remain outside its scope.
