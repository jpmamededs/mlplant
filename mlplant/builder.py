"""
builder.py: Orchestrates production project generation from parser output,
rendering Jinja2 templates into the output directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from mlplant.parser import parse_notebook

# Mapping: step -> generated Python filename under src/
STEP_TO_FILENAME = {
    "config": "config.py",
    "load_data": "data_loader.py",
    "preprocessing": "preprocessing.py",
    "features": "features.py",
    "train": "trainer.py",
    "evaluate": "evaluate.py",
    "artifacts": "artifacts.py",
    "predict": "predict.py",
}


@dataclass
class BuildOptions:
    notebook: str
    output: str = "./output"
    port: int = 8000
    title: str = "mlplant API"
    workers: int = 1
    docker: bool = False
    mlflow: bool = False
    extra: dict = field(default_factory=dict)


def _jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader("mlplant", "templates"),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_project(options: BuildOptions) -> Path:
    """
    Main entry point for project generation.
    Returns the Path of the generated output directory.
    """
    parse_result = parse_notebook(options.notebook)
    destination = Path(options.output)
    env = _jinja_env()

    base_context = {
        "title": options.title,
        "port": options.port,
        "workers": options.workers,
        "mlflow": options.mlflow,
        "detected_steps": parse_result.detected_steps,
    }

    # 1. Generate src/<file>.py for each annotated step
    for step, code in parse_result.blocks.items():
        filename = STEP_TO_FILENAME.get(step, f"{step}.py")
        template = env.get_template(f"{step}.py.j2")
        content = template.render(**base_context, code=code)
        _write_file(destination / "src" / filename, content)

    # 2. Generate main.py (FastAPI)
    api_template = env.get_template("main_api.py.j2")
    api_content = api_template.render(**base_context)
    _write_file(destination / "main.py", api_content)

    # 3. Generate requirements.txt
    requirements_template = env.get_template("requirements.txt.j2")
    requirements_content = requirements_template.render(**base_context)
    _write_file(destination / "requirements.txt", requirements_content)

    # 4. Dockerfile (optional)
    if options.docker:
        docker_template = env.get_template("Dockerfile.j2")
        docker_content = docker_template.render(**base_context)
        _write_file(destination / "Dockerfile", docker_content)

    return destination
