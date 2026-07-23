"""Typer command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nfl_coaching_sim.agents import (
    ExpectedPointsStrategy,
    ModelConfiguration,
    ModelProvider,
    MultiAgentStrategy,
    SingleAgentStrategy,
    make_model,
)
from nfl_coaching_sim.app import create_app
from nfl_coaching_sim.benchmark import (
    aggregate,
    read_results,
    run_benchmark,
    write_report,
    write_results,
)
from nfl_coaching_sim.data import (
    EVALUATION_SEASONS,
    TRAINING_SEASONS,
    build_scenarios,
    demo_scenarios,
    load_pbp,
    read_jsonl,
    rows_from_frame,
    write_jsonl,
    write_manifest,
)
from nfl_coaching_sim.simulator import DeterministicSimulator

app = typer.Typer(help="Multi-agent NFL late-game decision simulator.")
data_app = typer.Typer(help="Download and cache nflverse data.")
scenarios_app = typer.Typer(help="Build versioned scenario packs.")
simulator_app = typer.Typer(help="Train the deterministic evaluator.")
benchmark_app = typer.Typer(help="Run and report paired evaluations.")
ui_app = typer.Typer(help="Launch the Gradio application.")
app.add_typer(data_app, name="data")
app.add_typer(scenarios_app, name="scenarios")
app.add_typer(simulator_app, name="simulator")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(ui_app, name="app")


@data_app.command("sync")
def sync_data(
    output_dir: Annotated[Path, typer.Option()] = Path("data/cache"),
) -> None:
    """Download the fixed train/evaluation seasons as Parquet caches."""

    output_dir.mkdir(parents=True, exist_ok=True)
    training = load_pbp(TRAINING_SEASONS)
    evaluation = load_pbp(EVALUATION_SEASONS)
    training.write_parquet(output_dir / "pbp-training-2016-2023.parquet")
    evaluation.write_parquet(output_dir / "pbp-evaluation-2024-2025.parquet")
    typer.echo(f"Wrote nflverse caches to {output_dir}")


def _read_parquet(path: Path) -> list[dict]:
    import polars as pl

    return rows_from_frame(pl.read_parquet(path))


@scenarios_app.command("build")
def scenarios_build(
    training: Annotated[Path, typer.Option()] = Path(
        "data/cache/pbp-training-2016-2023.parquet"
    ),
    evaluation: Annotated[Path, typer.Option()] = Path(
        "data/cache/pbp-evaluation-2024-2025.parquet"
    ),
    output_dir: Annotated[Path, typer.Option()] = Path("data/scenarios"),
    limit: Annotated[int, typer.Option(min=25)] = 250,
) -> None:
    """Build benchmark and quick-start scenario packs from cached nflverse data."""

    scenarios = build_scenarios(_read_parquet(training), _read_parquet(evaluation), limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "benchmark-v1.jsonl"
    quick_path = output_dir / "quickstart-v1.jsonl"
    hashes = {
        benchmark_path.name: write_jsonl(scenarios, benchmark_path),
        quick_path.name: write_jsonl(scenarios[:25], quick_path),
    }
    write_manifest(output_dir / "manifest-v1.json", hashes, len(scenarios))
    typer.echo(f"Wrote {len(scenarios)} scenarios to {output_dir}")


@scenarios_app.command("demo")
def scenarios_demo(
    output: Annotated[Path, typer.Option()] = Path("data/scenarios/demo-v1.jsonl"),
) -> None:
    """Materialize the synthetic offline quick-start pack."""

    digest = write_jsonl(demo_scenarios(), output)
    typer.echo(f"Wrote 25 synthetic scenarios to {output} (sha256 {digest})")


@simulator_app.command("train")
def simulator_train(
    training: Annotated[Path, typer.Option()] = Path(
        "data/cache/pbp-training-2016-2023.parquet"
    ),
    output: Annotated[Path, typer.Option()] = Path("artifacts/simulator-v1.joblib"),
) -> None:
    """Train and save the fixed-seed action-value evaluator."""

    simulator = DeterministicSimulator().fit(_read_parquet(training))
    simulator.save(output)
    typer.echo(f"Wrote simulator artifact to {output}")


def _strategy(
    name: str,
    provider: str,
    model: str | None,
    upstream_url: str | None,
    model_license: str | None,
    base_url: str,
):
    if name == "expected_points":
        return ExpectedPointsStrategy()
    if not model or not upstream_url or not model_license:
        raise typer.BadParameter(
            "LLM strategies require --model, --upstream-url, and --model-license"
        )
    llm = make_model(
        ModelConfiguration(
            provider=ModelProvider(provider),
            model=model,
            upstream_url=upstream_url,
            license=model_license,
            base_url=base_url,
        )
    )
    return SingleAgentStrategy(llm) if name == "single_agent" else MultiAgentStrategy(llm)


@benchmark_app.command("run")
def benchmark_run(
    scenarios_path: Annotated[Path, typer.Option("--scenarios")] = Path(
        "data/scenarios/demo-v1.jsonl"
    ),
    output: Annotated[Path, typer.Option()] = Path("reports/results.jsonl"),
    strategies: Annotated[
        list[str], typer.Option("--strategy", help="Repeat for multiple strategies.")
    ] = ["expected_points"],
    simulator_path: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option()] = None,
    provider: Annotated[
        str, typer.Option(help="ollama or azure_foundry")
    ] = ModelProvider.OLLAMA.value,
    upstream_url: Annotated[str | None, typer.Option()] = None,
    model_license: Annotated[str | None, typer.Option()] = None,
    base_url: Annotated[str, typer.Option()] = "http://127.0.0.1:11434",
) -> None:
    """Run paired strategies over a scenario pack."""

    allowed = {"expected_points", "single_agent", "multi_agent"}
    unknown = set(strategies) - allowed
    if unknown:
        raise typer.BadParameter(f"unknown strategies: {sorted(unknown)}")
    scenario_rows = read_jsonl(scenarios_path)
    evaluator = (
        DeterministicSimulator.load(simulator_path)
        if simulator_path
        else DeterministicSimulator()
    )
    implementations = [
        _strategy(name, provider, model, upstream_url, model_license, base_url)
        for name in strategies
    ]
    results = run_benchmark(scenario_rows, implementations, evaluator)
    write_results(results, output)
    typer.echo(json.dumps(aggregate(results), indent=2))


@benchmark_app.command("report")
def benchmark_report(
    results: Annotated[Path, typer.Option()] = Path("reports/results.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("reports/benchmark.html"),
) -> None:
    """Create a self-contained HTML report."""

    write_report(read_results(results), output)
    typer.echo(f"Wrote report to {output}")


@ui_app.command("serve")
def app_serve(
    scenarios_path: Annotated[Path | None, typer.Option("--scenarios")] = None,
    simulator_path: Annotated[Path | None, typer.Option()] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 7860,
) -> None:
    """Launch the interactive Gradio application."""

    scenarios = read_jsonl(scenarios_path) if scenarios_path else demo_scenarios()
    evaluator = (
        DeterministicSimulator.load(simulator_path)
        if simulator_path
        else DeterministicSimulator()
    )
    create_app(scenarios, evaluator).launch(server_name=host, server_port=port)


if __name__ == "__main__":
    app()
