"""
builder.py: Orchestrates production project generation from parser output,
rendering Jinja2 templates into the output directory.
"""

from __future__ import annotations

import re as _re
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


# ---------------------------------------------------------------------------
# Notebook → API adaptation
# ---------------------------------------------------------------------------

# Imports that belong in trainer.py, not in other modules
_CLASSIFIER_IMPORT = _re.compile(
    r"^\s*(from\s+(xgboost|catboost|lightgbm)\b"
    r"|from\s+sklearn\.(?:model_selection\b.*\btrain_test_split|.*import.*(?:Classifier|Regressor)\b)"
    r"|import\s+(?:xgboost|catboost|lightgbm)\b)",
    _re.IGNORECASE,
)
# Notebook-specific model variable names that should not leak into API modules
_NOTEBOOK_MODEL_VAR = _re.compile(
    r"\b(catboost_model|xgbc_model|xgb_model|lgbm_model|rf_model)\b"
)
# `for df in [train_df, test_df]:` — notebook loop pattern in features cells
_FOR_DFS_LOOP = _re.compile(r"^\s*for\s+\w+\s+in\s+\[")


def _adapt_for_api(step: str, module_code: str, body_code: str):
    """Rewrite notebook-extracted code to be production API-compatible."""

    if step == "preprocessing":
        # Classifier imports live in trainer.py, not preprocessing
        cleaned_lines = [
            l for l in module_code.splitlines() if not _CLASSIFIER_IMPORT.match(l)
        ]
        module_code = "\n\n".join(
            chunk for chunk in "\n".join(cleaned_lines).split("\n\n") if chunk.strip()
        )
        # Adapt body: rename train_df/test_df → df; drop y/target/split lines
        _target_or_split = _re.compile(
            r"^\s*(y\s*=|target_col|X\s*=\s*\w+\s*$"
            r"|.*train_test_split"
            r"|X_train\s*,|X_val\s*,)"
        )
        lines, seen = [], set()
        for line in body_code.splitlines():
            if _target_or_split.match(line):
                continue
            line = _re.sub(r"\btrain_df\b", "df", line)
            line = _re.sub(r"\btest_df\b", "df", line)
            # Deduplicate identical consecutive lines (e.g., two identical get_dummies calls)
            if line.strip() and line in seen:
                continue
            seen.add(line)
            lines.append(line)
        body_code = "\n".join(lines)

    elif step == "features":
        # Unwrap `for df in [train_df, test_df]:` — body already uses `df` param
        lines = body_code.splitlines()
        if lines and _FOR_DFS_LOOP.match(lines[0]):
            inner = [l for l in lines[1:] if l.strip()]
            if inner:
                base_indent = min(len(l) - len(l.lstrip()) for l in inner)
                body_code = "\n".join(l[base_indent:] for l in inner if l.strip())

    elif step == "train":
        # Inject missing classifier/model imports that body_code references
        _KNOWN_IMPORTS = {
            "XGBClassifier": "from xgboost import XGBClassifier",
            "CatBoostClassifier": "from catboost import CatBoostClassifier",
            "LGBMClassifier": "from lightgbm import LGBMClassifier",
            "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
            "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
            "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
            "SVC": "from sklearn.svm import SVC",
            "XGBRegressor": "from xgboost import XGBRegressor",
            "CatBoostRegressor": "from catboost import CatBoostRegressor",
        }
        missing = [
            imp for cls, imp in _KNOWN_IMPORTS.items()
            if cls in body_code and imp not in module_code
        ]
        if missing:
            module_code = "\n".join(missing) + ("\n\n" + module_code if module_code else "")
        # Make eval_set conditional on X_val being provided
        body_code = _re.sub(
            r"eval_set\s*=\s*\[\(X_val,\s*y_val\)\]",
            "eval_set=[(X_val, y_val)] if X_val is not None else []",
            body_code,
        )
        body_code = _re.sub(
            r"eval_set\s*=\s*\(X_val,\s*y_val\)",
            "eval_set=(X_val, y_val) if X_val is not None else None",
            body_code,
        )
        # Keep CatBoost side artifacts under output/artifacts instead of workspace root.
        body_code = _re.sub(
            r"\bCatBoostClassifier\s*\(",
            "CatBoostClassifier(train_dir=os.path.join(os.getenv('MLPLANT_ARTIFACTS_DIR', 'artifacts'), 'catboost_info'), ",
            body_code,
        )
        body_code = _re.sub(
            r"\bCatBoostRegressor\s*\(",
            "CatBoostRegressor(train_dir=os.path.join(os.getenv('MLPLANT_ARTIFACTS_DIR', 'artifacts'), 'catboost_info'), ",
            body_code,
        )

    elif step == "evaluate":
        # Drop body that uses notebook-scoped model globals; template will use generic fallback
        if _NOTEBOOK_MODEL_VAR.search(body_code):
            body_code = ""

    elif step == "predict":
        # Batch-inference notebook code must not execute at import time in the API
        body_code = ""

    return module_code, body_code


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
    generated_steps = set()
    for step, code in parse_result.blocks.items():
        filename = STEP_TO_FILENAME.get(step, f"{step}.py")
        template = env.get_template(f"{step}.py.j2")
        module_code, body_code = split_module_body(code)
        module_code, body_code = _adapt_for_api(step, module_code, body_code)
        content = template.render(
            **base_context,
            code=code,
            module_code=module_code,
            body_code=body_code,
        )
        path = destination / "src" / filename
        _write_file(path, content)
        generated_steps.add(step)
        fire("post_step", step=step, path=path, content=content)

    # 1b. Always generate core pipeline modules (with safe defaults) so
    # build outputs are runnable even when some notebook annotations are absent.
    for step, filename in STEP_TO_FILENAME.items():
        if step in generated_steps:
            continue
        template = env.get_template(f"{step}.py.j2")
        content = template.render(
            **base_context,
            code="",
            module_code="",
            body_code="",
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

    # 4b. Generate train_pipeline.py
    train_pipeline_content = env.get_template("train_pipeline.py.j2").render(**base_context)
    _write_file(destination / "train_pipeline.py", train_pipeline_content)

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

