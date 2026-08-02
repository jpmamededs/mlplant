"""
cli.py: mlplant command-line interface powered by Typer.

Usage:
    mlplant build notebook.ipynb --output ./prod --port 8000 --docker --mlflow
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from mlplant import __version__
from mlplant.builder import BuildOptions, build_project
from mlplant.config_file import find_and_load
from mlplant.doctor import apply_fixes, analyze_notebook, export_fix_plan, merge_requirements, to_json
from mlplant.parser import parse_notebook

app = typer.Typer(
    name="mlplant",
    help=(
        "[bold cyan]mlplant[/bold cyan] — MLOps framework that converts annotated "
        "Jupyter notebooks into production-ready FastAPI projects.\n\n"
        "[dim]Annotate your cells with [bold]mlplant.config()[/bold], "
        "[bold]mlplant.train()[/bold], etc., then run [bold]build[/bold].[/dim]"
    ),
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mlplant [bold cyan]v{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    pass


@app.command("build")
def build(
    notebook: str = typer.Argument(..., help="Path to the [bold].ipynb[/bold] file."),
    output: str = typer.Option(
        "./output", "--output", "-o",
        help="Destination directory for the generated project.",
        rich_help_panel="Output",
    ),
    port: int = typer.Option(
        8000, "--port",
        help="Port the FastAPI/Uvicorn server will listen on.",
        rich_help_panel="Output",
    ),
    title: str = typer.Option(
        "mlplant API", "--title",
        help="Title shown in the Swagger UI.",
        rich_help_panel="Output",
    ),
    workers: int = typer.Option(
        1, "--workers",
        help="Number of Uvicorn worker processes.",
        rich_help_panel="Output",
    ),
    docker: bool = typer.Option(
        False, "--docker", is_flag=True,
        help="Generate an optimised [bold]Dockerfile[/bold] and .dockerignore.",
        rich_help_panel="Integrations",
    ),
    mlflow: bool = typer.Option(
        False, "--mlflow", is_flag=True,
        help="Inject [bold]MLflow[/bold] experiment tracking into the training step.",
        rich_help_panel="Integrations",
    ),
    ci: str = typer.Option(
        "", "--ci",
        help="Generate a CI/CD pipeline config. Choices: [bold]github[/bold] | [bold]gitlab[/bold].",
        rich_help_panel="Integrations",
        metavar="PLATFORM",
    ),
    ui: bool = typer.Option(
        False, "--ui", is_flag=True,
        help="Generate a modern [bold]React[/bold] frontend scaffold for /predict.",
        rich_help_panel="Integrations",
    ),
    train_after_build: bool = typer.Option(
        True,
        "--train/--no-train",
        help=(
            "Run generated train_pipeline.py right after build so model artifacts "
            "are ready for production API startup."
        ),
        rich_help_panel="Integrations",
    ),
    mode: str = typer.Option(
        "flex",
        "--mode",
        help="Build mode: flex (best-effort) or strict (fail on warnings).",
        rich_help_panel="Safety",
        show_default=True,
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        is_flag=True,
        help="Run a lightweight import/syntax smoke test against generated files.",
        rich_help_panel="Safety",
    ),
    emit_build_report: bool = typer.Option(
        True,
        "--build-report/--no-build-report",
        help="Generate mlplant_build_report.json with diagnostics and inferred decisions.",
        rich_help_panel="Safety",
    ),
):
    """Build a production [bold]FastAPI[/bold] project from an annotated notebook."""

    # Load mlplant.yaml and use its values as fallbacks for default CLI options
    yaml_cfg = find_and_load(Path(notebook).parent)
    if output == "./output":
        output = yaml_cfg.get("output", output)
    if port == 8000:
        port = int(yaml_cfg.get("port", port))
    if title == "mlplant API":
        title = yaml_cfg.get("title", title)
    if workers == 1:
        workers = int(yaml_cfg.get("workers", workers))
    if not docker:
        docker = bool(yaml_cfg.get("docker", docker))
    if not mlflow:
        mlflow = bool(yaml_cfg.get("mlflow", mlflow))
    if not ci:
        ci = yaml_cfg.get("ci", "") or ""
    if not ui:
        ui = bool(yaml_cfg.get("ui", ui))
    if mode == "flex":
        mode = str(yaml_cfg.get("mode", mode))
    if not smoke_test:
        smoke_test = bool(yaml_cfg.get("smoke_test", smoke_test))
    if emit_build_report:
        emit_build_report = bool(yaml_cfg.get("build_report", emit_build_report))

    mode = str(mode).strip().lower()
    if mode not in {"flex", "strict"}:
        console.print("[bold red]Error:[/bold red] --mode must be 'flex' or 'strict'.")
        raise typer.Exit(code=1)

    path = Path(notebook)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Notebook not found: {path}")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold cyan]mlplant build[/bold cyan]\n{path.name} -> {output}"))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
        task = progress.add_task("Parsing notebook...", total=None)
        try:
            parse_result = parse_notebook(notebook)
        except Exception as exc:
            console.print(f"[bold red]Parse error:[/bold red] {exc}")
            raise typer.Exit(code=1)
        progress.remove_task(task)

    if not parse_result.detected_steps and mode == "strict":
        console.print(
            "[bold red]Strict mode error:[/bold red] no [bold]@mlplant.*[/bold] annotations were found."
        )
        raise typer.Exit(code=1)

    if not parse_result.detected_steps:
        console.print("[yellow]Warning:[/yellow] No [bold]@mlplant.*[/bold] annotations found in the notebook.")
        choice = Prompt.ask(
            "Proceed with build anyway?",
            choices=["yes", "no"],
            default="no",
        )
        if choice == "no":
            console.print("[dim]Build cancelled.[/dim]")
            raise typer.Exit(code=0)

    options = BuildOptions(
        notebook=notebook,
        output=output,
        port=port,
        title=title,
        workers=workers,
        docker=docker,
        mlflow=mlflow,
        mlflow_tracking_uri=yaml_cfg.get("mlflow_tracking_uri", "http://localhost:5000"),
        ci=ci or None,
        ui=ui,
        mode=mode,
        emit_build_report=emit_build_report,
        project=yaml_cfg.get("project", "mlplant-api"),
        plugins=yaml_cfg.get("plugins", []),
    )

    steps_to_run = [
        ("Rendering pipeline steps...", None),
        ("Generating main API...", None),
        ("Writing requirements.txt...", None),
    ]
    if docker:
        steps_to_run.append(("Generating Dockerfile...", None))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
        task = progress.add_task("Building project...", total=None)
        try:
            destination = build_project(options)
        except Exception as exc:
            console.print(f"[bold red]Build error:[/bold red] {exc}")
            raise typer.Exit(code=1)
        progress.remove_task(task)

    console.print(f"\n[bold green]✓ Project successfully generated at:[/bold green] {destination.resolve()}")
    if ui:
        console.print("[bold green]✓ React UI generated at:[/bold green] "
                      f"{(Path(destination) / 'ui').resolve()}")

    report = options.extra.get("build_report") if isinstance(options.extra, dict) else None
    if isinstance(report, dict):
        warnings = report.get("warnings") or []
        if warnings:
            console.print("\n[yellow]Build warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  - {warning}")

    if smoke_test:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task("Running smoke test...", total=None)
            try:
                smoke_cmd = (
                    "import compileall; "
                    f"ok = compileall.compile_dir(r'{str(Path(destination).resolve())}', quiet=1); "
                    "print('ok' if ok else 'failed')"
                )
                smoke = subprocess.run(
                    [sys.executable, "-c", smoke_cmd],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                progress.remove_task(task)
                if exc.stdout:
                    console.print(exc.stdout.strip())
                if exc.stderr:
                    console.print(exc.stderr.strip())
                console.print("[bold red]Smoke test failed.[/bold red]")
                raise typer.Exit(code=1)

            progress.remove_task(task)
            output = (smoke.stdout or "").strip().lower()
            if "ok" in output:
                console.print("[bold green]✓ Smoke test passed.[/bold green]")
            else:
                console.print("[bold red]Smoke test failed.[/bold red]")
                raise typer.Exit(code=1)

    if train_after_build:
        notebook_base_dir = Path(notebook).resolve().parent
        output_dir = Path(destination).resolve()

        artifacts_dir = str((Path(destination) / "artifacts").resolve())
        train_env = os.environ.copy()
        train_env["MLPLANT_ARTIFACTS_DIR"] = artifacts_dir
        train_env.setdefault("MLFLOW_LOCAL_TRACKING_URI", f"sqlite:///{(output_dir / 'artifacts' / 'mlflow.db').resolve().as_posix()}")

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
            task = progress.add_task("Training generated pipeline...", total=None)
            try:
                completed = subprocess.run(
                    [sys.executable, str((Path(destination) / "train_pipeline.py").resolve())],
                    cwd=str(notebook_base_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                    env=train_env,
                )
            except subprocess.CalledProcessError as exc:
                progress.remove_task(task)
                if exc.stdout:
                    console.print("\n[bold]Training stdout:[/bold]")
                    console.print(exc.stdout.strip())
                if exc.stderr:
                    console.print("\n[bold]Training stderr:[/bold]")
                    console.print(exc.stderr.strip())

                stderr_lower = (exc.stderr or "").lower()
                missing_data_error = (
                    "filenotfounderror" in stderr_lower
                    or "no such file or directory" in stderr_lower
                )

                if missing_data_error:
                    console.print(
                        "\n[bold yellow]Training warning:[/bold yellow] "
                        "Auto-training was skipped because required dataset files "
                        "were not found."
                    )
                    console.print(
                        "[yellow]The project was generated successfully, but model artifacts "
                        "were not created yet. Update load_data() paths or provide the dataset, "
                        "then run train_pipeline.py manually.[/yellow]"
                    )
                    return

                console.print("\n[bold red]Training error:[/bold red] Generated pipeline failed to run.")
                raise typer.Exit(code=1)

            progress.remove_task(task)
            if completed.stdout:
                console.print("\n[bold]Training output:[/bold]")
                console.print(completed.stdout.strip())

        console.print("[bold green]✓ Artifacts generated. API is ready to serve predictions.[/bold green]")
    else:
        console.print("[yellow]Training skipped (--no-train). Run train_pipeline.py before serving the API.[/yellow]")


@app.command("inspect")
def inspect(
    notebook: str = typer.Argument(..., help="Path to the [bold].ipynb[/bold] file."),
):
    """Show the [bold]mlplant.*[/bold] steps detected in a notebook without generating any files."""
    path = Path(notebook)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Notebook not found: {path}")
        raise typer.Exit(code=1)

    result = parse_notebook(notebook)

    if not result.detected_steps:
        console.print("[yellow]No @mlplant.* annotations found in the notebook.[/yellow]")
        return

    console.print(Panel("[bold cyan]Detected steps[/bold cyan]"))
    for step in result.detected_steps:
        lines_count = len(result.blocks[step].splitlines())
        console.print(f"  [green]@mlplant.{step:<15}[/green] {lines_count} line(s)")


@app.command("doctor")
def doctor(
    notebook: str = typer.Argument(..., help="Path to the [bold].ipynb[/bold] file."),
    strict: bool = typer.Option(
        False,
        "--strict",
        is_flag=True,
        help="Return non-zero exit code when warnings or errors are found.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        is_flag=True,
        help="Print machine-readable JSON output.",
    ),
    export_fixes: str = typer.Option(
        "",
        "--export-fixes",
        help="Write a JSON fix plan to this path. If omitted, no file is written.",
    ),
    write_requirements: str = typer.Option(
        "",
        "--write-requirements",
        help="Merge inferred optional dependencies into the given requirements file.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        is_flag=True,
        help="Apply safe automatic fixes (uses dry-run by default).",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Preview fixes without writing files (default: dry-run).",
    ),
):
    """Run preflight diagnostics on a notebook before building."""
    path = Path(notebook)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Notebook not found: {path}")
        raise typer.Exit(code=1)

    try:
        report = analyze_notebook(path)
    except Exception as exc:
        console.print(f"[bold red]Doctor error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if as_json:
        console.print(to_json(report))
    else:
        summary = report.get("summary", {})
        console.print(Panel("[bold cyan]mlplant doctor[/bold cyan]"))
        console.print(f"Notebook: {report.get('notebook')}")
        console.print("Detected steps: " + ", ".join(report.get("detected_steps", [])) if report.get("detected_steps") else "Detected steps: none")
        console.print(
            "Summary: "
            f"errors={summary.get('errors', 0)} "
            f"warnings={summary.get('warnings', 0)} "
            f"info={summary.get('info', 0)}"
        )

        optional_deps = report.get("inferred_optional_dependencies", [])
        if optional_deps:
            console.print("\n[bold]Inferred optional dependencies:[/bold]")
            for dep in optional_deps:
                console.print(f"  - {dep}")

        issues = report.get("issues", [])
        if issues:
            console.print("\n[bold]Issues:[/bold]")
            for issue in issues:
                sev = (issue.get("severity") or "info").upper()
                code = issue.get("code", "unknown")
                msg = issue.get("message", "")
                hint = issue.get("hint", "")
                console.print(f"  {sev} {code}: {msg}")
                if hint:
                    console.print(f"    hint: {hint}")
        else:
            console.print("\n[bold green]No issues detected.[/bold green]")

    if export_fixes:
        try:
            exported_path = export_fix_plan(report, export_fixes)
        except Exception as exc:
            console.print(f"[bold red]Doctor export error:[/bold red] {exc}")
            raise typer.Exit(code=1)
        console.print(f"[bold green]✓ Fix plan exported:[/bold green] {exported_path}")

    if write_requirements:
        inferred = report.get("inferred_optional_dependencies", []) or []
        try:
            merge_result = merge_requirements(write_requirements, inferred)
        except Exception as exc:
            console.print(f"[bold red]Doctor requirements merge error:[/bold red] {exc}")
            raise typer.Exit(code=1)

        added = merge_result.get("added", [])
        console.print(
            f"[bold green]✓ Requirements updated:[/bold green] {merge_result.get('path')}"
        )
        if added:
            console.print("Added dependencies:")
            for dep in added:
                console.print(f"  - {dep}")
        else:
            console.print("No new dependencies were added.")

    if apply:
        requirements_target = write_requirements or ""
        if not requirements_target:
            default_requirements = Path(path).resolve().parent / "requirements.txt"
            if default_requirements.exists():
                requirements_target = str(default_requirements)

        try:
            apply_result = apply_fixes(
                notebook_path=path,
                report=report,
                dry_run=dry_run,
                requirements_path=requirements_target or None,
            )
        except Exception as exc:
            console.print(f"[bold red]Doctor apply error:[/bold red] {exc}")
            raise typer.Exit(code=1)

        changes = apply_result.get("changes", {})
        mode_label = "DRY-RUN" if dry_run else "APPLIED"
        console.print(f"\n[bold cyan]Doctor apply ({mode_label})[/bold cyan]")
        console.print(
            "Notebook changes: "
            f"normalized_install_lines={changes.get('normalized_plain_install_lines', 0)} "
            f"stub_cells={changes.get('added_stub_cells', 0)}"
        )

        missing_stubs = changes.get("missing_core_step_stubs", []) or []
        if missing_stubs:
            console.print("Core step stubs: " + ", ".join(missing_stubs))

        req_result = apply_result.get("requirements")
        if isinstance(req_result, dict):
            console.print(f"Requirements target: {req_result.get('path')}")
            added = req_result.get("added", []) or []
            if added:
                console.print("Requirements additions:")
                for dep in added:
                    console.print(f"  - {dep}")

        if dry_run:
            console.print(
                "[yellow]Dry-run only:[/yellow] rerun with [bold]--no-dry-run[/bold] to persist changes."
            )

    errors = int(report.get("summary", {}).get("errors", 0))
    warnings = int(report.get("summary", {}).get("warnings", 0))
    if strict and (errors > 0 or warnings > 0):
        raise typer.Exit(code=1)
