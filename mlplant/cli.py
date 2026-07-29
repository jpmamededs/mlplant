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

from mlplant.builder import BuildOptions, build_project

app = typer.Typer(
    name="mlplant",
    help="Converts Jupyter notebooks into production-ready API projects.",
    add_completion=False,
)

console = Console()


@app.command("build")
def build(
    notebook: str = typer.Argument(..., help="Path to the .ipynb file"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    port: int = typer.Option(8000, "--port", help="FastAPI server port"),
    title: str = typer.Option("mlplant API", "--title", help="Swagger UI title"),
    workers: int = typer.Option(1, "--workers", help="Number of Uvicorn workers"),
    docker: bool = typer.Option(False, "--docker", is_flag=True, help="Generate an optimized Dockerfile"),
    mlflow: bool = typer.Option(False, "--mlflow", is_flag=True, help="Inject MLflow tracking into the training step"),
):
    """Build a production project from an annotated notebook."""

    path = Path(notebook)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Notebook not found: {path}")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold cyan]mlplant build[/bold cyan]\n{path.name} -> {output}"))

    options = BuildOptions(
        notebook=notebook,
        output=output,
        port=port,
        title=title,
        workers=workers,
        docker=docker,
        mlflow=mlflow,
    )

    try:
        destination = build_project(options)
        console.print(f"\n[bold green]✓ Project successfully generated at:[/bold green] {destination.resolve()}")
    except Exception as exc:
        console.print(f"[bold red]Build error:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command("inspect")
def inspect(
    notebook: str = typer.Argument(..., help="Path to the .ipynb file"),
):
    """List the @mlplant.* steps found in the notebook without generating files."""
    from mlplant.parser import parse_notebook

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
