"""
cli.py: mlplant command-line interface powered by Typer.

Usage:
    mlplant build notebook.ipynb --output ./prod --port 8000 --docker --mlflow
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from mlplant import __version__
from mlplant.builder import BuildOptions, build_project
from mlplant.config_file import find_and_load
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
