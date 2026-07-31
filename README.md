# Multi-Agent NFL Coaching Simulator

An open-source research application for asking whether a deliberating NFL coaching staff outperforms a single language-model coach or a simple expected-points policy. Five
role agents independently recommend a call, review anonymized arguments, revise their positions, and send the debate to a head-coach synthesizer. A deterministic, held-out
action-value model scores the result.

The project is local-first. Its default installation includes Gradio, LangChain, Ollama, Azure AI Foundry authentication, scikit-learn, Polars, and nflverse data. The
application does not select or download a model for you.

## What is included

- A Gradio scenario explorer with stage-by-stage deliberation updates.
- A deterministic situation brief covering score, clock, field-goal distance, field zone, timeout leverage, and endgame first-down value.
- Evidence-linked action scorecards requiring every coach to compare every legal call, name the closest alternative, and define a switch condition.
- A Typer CLI for nflverse sync, scenario construction, evaluator training, paired benchmarks, and HTML reports.
- A 40-case synthetic offline pack for trying the UI and harness without downloading NFL data.
- A reproducible builder for the CC-BY-4.0 40- and 250-scenario nflverse release packs.
- Fixed-seed WPA/EPA evaluation, an intentionally simpler bucketed EPA baseline, and paired bootstrap confidence intervals.

## Quick start

Use Python 3.12 or 3.13:

```shell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\nfl-coach
```

On macOS or Linux, use `.venv/bin/` instead of `.venv\Scripts\`.

With uv, synchronize once and skip repeat environment checks on normal launches:

```shell
uv sync --python 3.13
uv run --no-sync nfl-coach
```

Run a normal `uv sync` again whenever `pyproject.toml` or `uv.lock` changes.

The bare `nfl-coach` command launches Gradio with the checked-in nflverse quick-start scenarios. It automatically uses `artifacts/simulator-v1.joblib` when present and
otherwise uses the deterministic offline evaluator.

Use `nfl-coach serve --help` when you need a custom host, port, scenario pack, or simulator path. `python -m nfl_coaching_sim` is an equivalent startup command.

The expected-points strategy works immediately. For an Ollama strategy, start an Ollama server and enter all of the following in the UI:

- The installed Ollama model tag.
- Optionally, the model's upstream project or weight URL.
- Its approved SPDX license.
- The Ollama base URL.

The included software does not claim that an arbitrary Ollama tag is open source. Users remain responsible for accurately declaring the selected weights' license.

### Azure AI Foundry

Choose **Azure AI Foundry** in Gradio or pass `--provider azure_foundry` to the benchmark command. Interactive use requires:

- The Foundry deployment name as the model.
- The deployment endpoint ending in `/openai/v1/`.
- Its approved open-model SPDX license.

The model source URL is optional in Gradio because it is not used for inference. CLI LLM benchmarks require it so exported research results retain model provenance.

Authentication uses `DefaultAzureCredential`, so local Azure CLI credentials, managed identity, workload identity, and environment-based service principals work without
putting secrets in the app. If key authentication is unavoidable, set
`AZURE_FOUNDRY_API_KEY` in the process environment. Keys are never accepted through Gradio or CLI options and are not written to benchmark artifacts.
Foundry model calls use LangChain's Responses API mode (`use_responses_api=True`). Each provider adapter allowlists its supported generation parameters: Foundry Responses receives `temperature`, while Ollama receives `temperature` and `seed`.

Example:

```shell
nfl-coach benchmark run ^
  --strategy single_agent ^
  --provider azure_foundry ^
  --model YOUR_FOUNDRY_DEPLOYMENT ^
  --base-url https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/ ^
  --upstream-url https://example.org/upstream-open-model ^
  --model-license Apache-2.0
```

Azure AI Foundry is a proprietary hosted provider. The application continues to enforce an approved open license for the selected model, and benchmark provenance distinguishes
the serving provider from the model license.

### LangChain provider boundary

All coaching strategies depend on one provider-neutral structured-model interface. Provider adapters under `nfl_coaching_sim.providers` own endpoint validation, authentication,
supported parameters, API mode, and lazy LangChain model construction. Structured-output repair, prompts, orchestration, and fallback behavior remain shared. Registered
providers automatically appear in Gradio, and the CLI accepts any registered provider ID.

Add a LangChain-compatible provider by implementing the `ProviderAdapter` protocol and registering one adapter during startup:

```python
from nfl_coaching_sim.providers import register_provider

register_provider(MyProviderAdapter())
```

The adapter declares `ProviderCapabilities`, validates its own configuration, and returns a `ProviderModel`. A function-based `register_model_provider(...)` compatibility helper
remains available for simple integrations. No coaching strategy, prompt, benchmark, or UI callback needs provider-specific logic.

## Rebuild the research artifacts

The released scenarios work without setup. To download nflverse play-by-play, regenerate both scenario packs, and train the simulator in one step:

```shell
nfl-coach setup
```

Then run a benchmark and report:

```shell
nfl-coach benchmark run --scenarios data/scenarios/quickstart-v1.jsonl
nfl-coach benchmark report
```

The granular `data sync`, `scenarios build`, and `simulator train` commands remain available when only one artifact needs rebuilding.

To include model strategies, repeat `--strategy` and provide model provenance:

```shell
nfl-coach benchmark run ^
  --strategy expected_points ^
  --strategy single_agent ^
  --strategy multi_agent ^
  --model YOUR_OLLAMA_TAG ^
  --upstream-url https://example.org/upstream-model ^
  --model-license Apache-2.0
```

Use shell-appropriate line continuation on non-Windows platforms.

## Evaluation design

Training uses the 2016–2023 seasons. Released evaluation scenarios use 2024–2025 and are limited to regulation plays in quarters three and four, offense win probability from
10% through 90%, and score differential within 16 points. Nullified plays, kneels, spikes, and incomplete states are excluded.

The primary score is expected win-probability added. EPA, oracle regret, best-action rate, failures, fallbacks, latency, and model-call counts are secondary outputs. Neither
agents nor the EP policy see the richer evaluator's counterfactual scores.

Prompt version 2 builds an outcome-free situation brief from each released scenario. Specialist coaches receive the same evidence packet as the single-agent head coach, but
use role-specific checklists. Multi-agent recommendations must assess every legal action and may cite only evidence included in that packet.

This is an observational benchmark. The action-value regressors learn from choices NFL coaches actually made, so their counterfactual scores are not causal estimates and
retain selection bias. Results should be described as performance against this released simulator—not proof that an unobserved call would have produced the modeled outcome.

## Data and licenses

Code is Apache-2.0. nflverse-derived scenario packs are CC-BY-4.0 and include a source manifest and hashes. See [data/README.md](data/README.md) for attribution and the public
artifact schema. Raw nflverse data and model weights are never committed.

## Development

The retained test suite has five critical-path tests:

```shell
python -m pip install -e ".[test]"
pytest
```

It covers data isolation, deterministic scoring, full deliberation/fallback, paired benchmarking, and Gradio construction/callbacks. Temporary scripts, snapshots, generated
reports, and low-value implementation-detail tests are not kept in the repository.
