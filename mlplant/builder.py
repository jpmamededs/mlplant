"""
builder.py: Orchestrates production project generation from parser output,
rendering Jinja2 templates into the output directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, PackageLoader, select_autoescape

from mlplant.parser import parse_notebook, split_module_body

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
    mlflow_tracking_uri: str = "http://localhost:5000"
    ci: Optional[str] = None                        # "github" | "gitlab" | None
    project: str = "mlplant-api"
    plugins: List[str] = field(default_factory=list)
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
    from mlplant.plugins import fire, load as load_plugins
    from mlplant.schema_gen import infer_fields

    if options.plugins:
        load_plugins(options.plugins)

    parse_result = parse_notebook(options.notebook)
    destination = Path(options.output)
    env = _jinja_env()

    base_context = {
        "project": options.project,
        "title": options.title,
        "port": options.port,
        "workers": options.workers,
        "mlflow": options.mlflow,
        "mlflow_tracking_uri": options.mlflow_tracking_uri,
        "docker": options.docker,
        "detected_steps": parse_result.detected_steps,
    }

    fire("pre_build", options=options, parse_result=parse_result)

    # 1. Generate src/<file>.py for each annotated step
    for step, code in parse_result.blocks.items():
        filename = STEP_TO_FILENAME.get(step, f"{step}.py")
        template = env.get_template(f"{step}.py.j2")
        module_code, body_code = split_module_body(code)
        content = template.render(
            **base_context,
            code=code,
            module_code=module_code,
            body_code=body_code,
        )
        path = destination / "src" / filename
        _write_file(path, content)
        fire("post_step", step=step, path=path, content=content)

    # 2. Generate main.py (FastAPI)
    api_content = env.get_template("main_api.py.j2").render(**base_context)
    _write_file(destination / "main.py", api_content)

    # 3. Generate src/schemas.py (Pydantic data contract)
    fields = infer_fields(parse_result)
    schemas_content = env.get_template("schemas.py.j2").render(**base_context, fields=fields)
    _write_file(destination / "src" / "schemas.py", schemas_content)

    # 4. Generate requirements.txt
    req_content = env.get_template("requirements.txt.j2").render(**base_context)
    _write_file(destination / "requirements.txt", req_content)

    # 5. Dockerfile (optional)
    if options.docker:
        docker_content = env.get_template("Dockerfile.j2").render(**base_context)
        _write_file(destination / "Dockerfile", docker_content)

    # 6. CI/CD config (optional)
    if options.ci == "github":
        ci_content = env.get_template("github_workflow.yaml.j2").render(**base_context)
        _write_file(destination / ".github" / "workflows" / "mlplant-ci.yaml", ci_content)
    elif options.ci == "gitlab":
        ci_content = env.get_template("gitlab_ci.yaml.j2").render(**base_context)
        _write_file(destination / ".gitlab-ci.yml", ci_content)

    fire("post_build", destination=destination, options=options)

    return destination

