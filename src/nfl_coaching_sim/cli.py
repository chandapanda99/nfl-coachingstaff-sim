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

app = typer.Typer(
    help="Virtual NFL coaching staff and late-game decision lab.",
    invoke_without_command=True,
    no_args_is_help=False,
)
data_app = typer.Typer(help="Load and cache nflverse game film.")
scenarios_app = typer.Typer(help="Chart versioned late-game situation packs.")
simulator_app = typer.Typer(help="Calibrate the deterministic decision grader.")
benchmark_app = typer.Typer(help="Grade calls and produce postgame reports.")
ui_app = typer.Typer(help="Open the virtual sideline.")
app.add_typer(data_app, name="data")
app.add_typer(scenarios_app, name="scenarios")
app.add_typer(simulator_app, name="simulator")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(ui_app, name="app")

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_SCENARIO_DIR = Path("data/scenarios")
DEFAULT_QUICKSTART_PATH = DEFAULT_SCENARIO_DIR / "quickstart-v1.jsonl"
DEFAULT_SIMULATOR_PATH = Path("artifacts/simulator-v1.joblib")


def _sync_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo("Pulling historical nflverse game film for the analytics staff…")
    training = load_pbp(TRAINING_SEASONS)
    typer.echo("Pulling held-out seasons for the postgame grading room…")
    evaluation = load_pbp(EVALUATION_SEASONS)
    training.write_parquet(output_dir / "pbp-training-2016-2023.parquet")
    evaluation.write_parquet(output_dir / "pbp-evaluation-2024-2025.parquet")


def _build_scenario_packs(
    training: Path,
    evaluation: Path,
    output_dir: Path,
    limit: int,
) -> None:
    scenarios = build_scenarios(_read_parquet(training), _read_parquet(evaluation), limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "benchmark-v1.jsonl"
    quick_path = output_dir / "quickstart-v1.jsonl"
    hashes = {
        benchmark_path.name: write_jsonl(scenarios, benchmark_path),
        quick_path.name: write_jsonl(scenarios[:25], quick_path),
    }
    write_manifest(output_dir / "manifest-v1.json", hashes, len(scenarios))


def _train_simulator(training: Path, output: Path) -> None:
    simulator = DeterministicSimulator().fit(_read_parquet(training))
    simulator.save(output)


def _load_startup_assets(
    scenarios_path: Path | None,
    simulator_path: Path | None,
):
    if scenarios_path is not None:
        scenarios = read_jsonl(scenarios_path)
    elif DEFAULT_QUICKSTART_PATH.exists():
        scenarios = read_jsonl(DEFAULT_QUICKSTART_PATH)
    else:
        scenarios = demo_scenarios()

    resolved_simulator = simulator_path
    if resolved_simulator is None and DEFAULT_SIMULATOR_PATH.exists():
        resolved_simulator = DEFAULT_SIMULATOR_PATH
    evaluator = DeterministicSimulator.load(resolved_simulator) if resolved_simulator is not None else DeterministicSimulator()
    return scenarios, evaluator


def _launch_app(
    scenarios_path: Path | None = None,
    simulator_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    scenarios, evaluator = _load_startup_assets(scenarios_path, simulator_path)
    create_app(scenarios, evaluator).launch(server_name=host, server_port=port)


@app.callback()
def main(ctx: typer.Context) -> None:
    """Launch the app when no subcommand is supplied."""

    if ctx.invoked_subcommand is None:
        _launch_app()


@data_app.command("sync")
def sync_data(
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_CACHE_DIR,
) -> None:
    """Download the fixed train/evaluation seasons as Parquet caches."""

    _sync_data(output_dir)
    typer.echo(f"Film-room cache is ready at {output_dir}")


def _read_parquet(path: Path) -> list[dict]:
    import polars as pl

    return rows_from_frame(pl.read_parquet(path))


@scenarios_app.command("build")
def scenarios_build(
    training: Annotated[Path, typer.Option()] = Path("data/cache/pbp-training-2016-2023.parquet"),
    evaluation: Annotated[Path, typer.Option()] = Path("data/cache/pbp-evaluation-2024-2025.parquet"),
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_SCENARIO_DIR,
    limit: Annotated[int, typer.Option(min=25)] = 250,
) -> None:
    """Build benchmark and quick-start scenario packs from cached nflverse data."""

    _build_scenario_packs(training, evaluation, output_dir, limit)
    typer.echo(f"Charted {limit} late-game situations in {output_dir}")


@scenarios_app.command("demo")
def scenarios_demo(
    output: Annotated[Path, typer.Option()] = Path("data/scenarios/demo-v1.jsonl"),
) -> None:
    """Materialize the synthetic offline quick-start pack."""

    digest = write_jsonl(demo_scenarios(), output)
    typer.echo(f"Created 25 preseason walkthrough situations at {output} (sha256 {digest})")


@simulator_app.command("train")
def simulator_train(
    training: Annotated[Path, typer.Option()] = Path("data/cache/pbp-training-2016-2023.parquet"),
    output: Annotated[Path, typer.Option()] = DEFAULT_SIMULATOR_PATH,
) -> None:
    """Train and save the fixed-seed action-value evaluator."""

    _train_simulator(training, output)
    typer.echo(f"Decision-grading model is ready at {output}")


@app.command("setup")
def setup(
    cache_dir: Annotated[Path, typer.Option()] = DEFAULT_CACHE_DIR,
    scenario_dir: Annotated[Path, typer.Option()] = DEFAULT_SCENARIO_DIR,
    simulator_path: Annotated[Path, typer.Option()] = DEFAULT_SIMULATOR_PATH,
) -> None:
    """Download nflverse data and rebuild all local research artifacts."""

    training = cache_dir / "pbp-training-2016-2023.parquet"
    evaluation = cache_dir / "pbp-evaluation-2024-2025.parquet"
    _sync_data(cache_dir)
    typer.echo("Charting down, distance, field position, score, and clock situations…")
    _build_scenario_packs(training, evaluation, scenario_dir, 250)
    typer.echo("Calibrating the postgame decision grader…")
    _train_simulator(training, simulator_path)
    typer.echo("The staff room is ready. Run `nfl-coach` and send in the first call.")


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
            "LLM benchmark exports require --model, --upstream-url, and " "--model-license for reproducible provenance"
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
    scenarios_path: Annotated[Path, typer.Option("--scenarios")] = Path("data/scenarios/demo-v1.jsonl"),
    output: Annotated[Path, typer.Option()] = Path("reports/results.jsonl"),
    strategies: Annotated[list[str], typer.Option("--strategy", help="Repeat for multiple strategies.")] = ["expected_points"],
    simulator_path: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option()] = None,
    provider: Annotated[str, typer.Option(help="ollama or azure_foundry")] = ModelProvider.OLLAMA.value,
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
    evaluator = DeterministicSimulator.load(simulator_path) if simulator_path else DeterministicSimulator()
    implementations = [_strategy(name, provider, model, upstream_url, model_license, base_url) for name in strategies]
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
    typer.echo(f"Postgame decision report is ready at {output}")


@ui_app.command("serve")
def app_serve(
    scenarios_path: Annotated[Path | None, typer.Option("--scenarios")] = None,
    simulator_path: Annotated[Path | None, typer.Option()] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 7860,
) -> None:
    """Launch the interactive Gradio application."""

    _launch_app(scenarios_path, simulator_path, host, port)


@app.command("serve")
def serve(
    scenarios_path: Annotated[Path | None, typer.Option("--scenarios")] = None,
    simulator_path: Annotated[Path | None, typer.Option()] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 7860,
) -> None:
    """Launch Gradio with optional custom assets and network settings."""

    _launch_app(scenarios_path, simulator_path, host, port)


if __name__ == "__main__":
    app()
