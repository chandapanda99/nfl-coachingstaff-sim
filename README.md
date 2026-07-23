# Multi-Agent NFL Coaching Simulator

An open-source research application for asking whether a deliberating NFL coaching
staff outperforms a single language-model coach or a simple expected-points policy.
Five role agents independently recommend a call, review anonymized arguments, revise
their positions, and send the debate to a head-coach synthesizer. A deterministic,
held-out action-value model scores the result.

The project is local-first. It uses Gradio, LangChain, Ollama, scikit-learn,
Polars, and nflverse data, with optional Azure AI Foundry model serving. Ollama
remains the default, and the application does not select or download a model for you.

## What is included

- A Gradio scenario explorer with stage-by-stage deliberation updates.
- A Typer CLI for nflverse sync, scenario construction, evaluator training, paired
  benchmarks, and HTML reports.
- A 25-case synthetic offline pack for trying the UI and harness without downloading
  NFL data.
- A reproducible builder for the CC-BY-4.0 25- and 250-scenario nflverse release packs.
- Fixed-seed WPA/EPA evaluation, an intentionally simpler bucketed EPA baseline, and
  paired bootstrap confidence intervals.

## Install

Use Python 3.12 or 3.13:

```shell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

On macOS or Linux, use `.venv/bin/python` in place of the Windows path.
To enable Azure AI Foundry, install `".[test,azure]"` instead.

Materialize the offline demo pack and launch Gradio:

```shell
nfl-coach scenarios demo
nfl-coach app serve --scenarios data/scenarios/demo-v1.jsonl
```

The expected-points strategy works immediately. For an Ollama strategy, start an
Ollama server and enter all of the following in the UI:

- The installed Ollama model tag.
- The model's upstream project or weight URL.
- Its approved SPDX license.
- The Ollama base URL.

The included software does not claim that an arbitrary Ollama tag is open source.
Users remain responsible for accurately declaring the selected weights' license.

### Azure AI Foundry

Choose **Azure AI Foundry** in Gradio or pass `--provider azure_foundry` to the
benchmark command. Supply:

- The Foundry deployment name as the model.
- The deployment endpoint ending in `/openai/v1/`.
- The upstream model project URL and its approved open-model SPDX license.

Authentication uses `DefaultAzureCredential`, so local Azure CLI credentials,
managed identity, workload identity, and environment-based service principals work
without putting secrets in the app. If key authentication is unavoidable, set
`AZURE_FOUNDRY_API_KEY` in the process environment. Keys are never accepted through
Gradio or CLI options and are not written to benchmark artifacts.

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

Azure AI Foundry is a proprietary hosted provider. The application continues to
enforce an approved open license for the selected model, and benchmark provenance
distinguishes the serving provider from the model license.

## Build the research artifacts

```shell
nfl-coach data sync
nfl-coach scenarios build
nfl-coach simulator train
nfl-coach benchmark run --scenarios data/scenarios/quickstart-v1.jsonl
nfl-coach benchmark report
```

To include local-model strategies, repeat `--strategy` and provide model provenance:

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

Training uses the 2016–2023 seasons. Released evaluation scenarios use 2024–2025 and
are limited to regulation plays in quarters three and four, offense win probability
from 10% through 90%, and score differential within 16 points. Nullified plays,
kneels, spikes, and incomplete states are excluded.

The primary score is expected win-probability added. EPA, oracle regret, best-action
rate, failures, fallbacks, latency, and model-call counts are secondary outputs.
Neither agents nor the EP policy see the richer evaluator's counterfactual scores.

This is an observational benchmark. The action-value regressors learn from choices
NFL coaches actually made, so their counterfactual scores are not causal estimates
and retain selection bias. Results should be described as performance against this
released simulator—not proof that an unobserved call would have produced the modeled
outcome.

## Data and licenses

Code is Apache-2.0. nflverse-derived scenario packs are CC-BY-4.0 and include a
source manifest and hashes. See [data/README.md](data/README.md) for attribution and
the public artifact schema. Raw nflverse data and model weights are never committed.

## Development

The retained test suite has five critical-path tests:

```shell
pytest
```

It covers data isolation, deterministic scoring, full deliberation/fallback,
paired benchmarking, and Gradio construction/callbacks. Temporary scripts,
snapshots, generated reports, and low-value implementation-detail tests are not
kept in the repository.
