"""
builder.py: Orchestrates production project generation from parser output,
rendering Jinja2 templates into the output directory.
"""

from __future__ import annotations

import ast
import json
import re as _re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

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

# Optional dependencies inferred from notebook imports/usages.
_MODULE_REQUIREMENTS: Dict[str, str] = {
    "xgboost": "xgboost>=2.0",
    "catboost": "catboost>=1.2",
    "lightgbm": "lightgbm>=4.3",
    "imblearn": "imbalanced-learn>=0.12",
    "imbalanced_learn": "imbalanced-learn>=0.12",
    "category_encoders": "category-encoders>=2.6",
    "feature_engine": "feature-engine>=1.8",
    "optuna": "optuna>=3.6",
    "shap": "shap>=0.45",
    "scipy": "scipy>=1.11",
    "statsmodels": "statsmodels>=0.14",
    "tensorflow": "tensorflow>=2.16",
    "torch": "torch>=2.2",
    "polars": "polars>=1.0",
    "pyarrow": "pyarrow>=16.0",
}

_SYMBOL_REQUIREMENTS: Dict[str, str] = {
    "XGBClassifier": "xgboost>=2.0",
    "XGBRegressor": "xgboost>=2.0",
    "CatBoostClassifier": "catboost>=1.2",
    "CatBoostRegressor": "catboost>=1.2",
    "LGBMClassifier": "lightgbm>=4.3",
    "LGBMRegressor": "lightgbm>=4.3",
    "SMOTE": "imbalanced-learn>=0.12",
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
    ui: bool = False
    mode: str = "flex"  # "flex" | "strict"
    emit_build_report: bool = True
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


def _example_value(field_name: str, python_type: str):
    lname = field_name.lower()

    if python_type == "str":
        if "city" in lname or "cidade" in lname:
            return "sao_paulo"
        if "state" in lname or "estado" in lname:
            return "sp"
        if "gender" in lname or "sexo" in lname:
            return "male"
        return "sample"

    if python_type == "bool":
        return False

    if python_type == "int":
        return 1

    return 0.0


def _build_request_example(fields) -> dict:
    return {f.name: _example_value(f.name, f.python_type) for f in fields}


def _find_logo_file(notebook_path: str) -> Optional[Path]:
    notebook_dir = Path(notebook_path).resolve().parent
    candidates = [
        notebook_dir / "assets" / "mlplant.svg",
        notebook_dir.parent / "assets" / "mlplant.svg",
        Path(__file__).resolve().parent.parent / "assets" / "mlplant.svg",
        Path.cwd() / "assets" / "mlplant.svg",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def _generate_react_ui(env: Environment, destination: Path, base_context: dict, notebook_path: str) -> None:
    ui_context = {
        **base_context,
        "api_base": f"http://localhost:{base_context['port']}",
    }

    ui_files = {
        "ui_package.json.j2": destination / "ui" / "package.json",
        "ui_index.html.j2": destination / "ui" / "index.html",
        "ui_vite.config.js.j2": destination / "ui" / "vite.config.js",
        "ui_src_main.jsx.j2": destination / "ui" / "src" / "main.jsx",
        "ui_src_app.jsx.j2": destination / "ui" / "src" / "App.jsx",
        "ui_src_styles.css.j2": destination / "ui" / "src" / "styles.css",
        "ui_readme.md.j2": destination / "ui" / "README.md",
    }

    for template_name, output_path in ui_files.items():
        content = env.get_template(template_name).render(**ui_context)
        _write_file(output_path, content)

    logo_source = _find_logo_file(notebook_path)
    logo_target = destination / "ui" / "public" / "mlplant.svg"
    logo_target.parent.mkdir(parents=True, exist_ok=True)

    if logo_source is not None:
        shutil.copyfile(logo_source, logo_target)
    else:
        logo_target.write_text(
            """<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 320 80\"><rect width=\"320\" height=\"80\" rx=\"16\" fill=\"#111827\"/><text x=\"24\" y=\"50\" fill=\"#f8fafc\" font-size=\"28\" font-family=\"Arial, sans-serif\">mlplant</text></svg>""",
            encoding="utf-8",
        )


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
            "CatBoostClassifier(train_dir=_resolve_catboost_train_dir(), ",
            body_code,
        )
        body_code = _re.sub(
            r"\bCatBoostRegressor\s*\(",
            "CatBoostRegressor(train_dir=_resolve_catboost_train_dir(), ",
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


def _detect_extra_requirements(parse_result) -> List[str]:
    """Infer optional pip dependencies from notebook-imported code."""
    requirements: Set[str] = set()
    combined_code = "\n\n".join(parse_result.blocks.values())

    try:
        tree = ast.parse(combined_code) if combined_code.strip() else None
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    req = _MODULE_REQUIREMENTS.get(root)
                    if req:
                        requirements.add(req)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                req = _MODULE_REQUIREMENTS.get(root)
                if req:
                    requirements.add(req)

    for symbol, req in _SYMBOL_REQUIREMENTS.items():
        if symbol in combined_code:
            requirements.add(req)

    return sorted(requirements)


def _has_absolute_path_reference(code: str) -> bool:
    """Detect hardcoded absolute paths that reduce portability."""
    windows_abs = _re.search(r"[A-Za-z]:[\\/][^\n'\"]+", code)
    unix_abs = _re.search(r"(^|[\s(=])/[\w./-]+", code)
    return bool(windows_abs or unix_abs)


def _build_warnings(parse_result, extra_requirements: List[str]) -> List[str]:
    warnings: List[str] = []

    diagnostics = getattr(parse_result, "diagnostics", []) or []
    warnings.extend(diagnostics)

    if not parse_result.detected_steps:
        warnings.append("No annotated mlplant steps were detected in the notebook.")

    if "predict" not in parse_result.detected_steps:
        warnings.append("No predict step was detected; generated API may not serve domain-specific predictions.")

    combined_code = "\n\n".join(parse_result.blocks.values())
    if combined_code and _has_absolute_path_reference(combined_code):
        warnings.append("Absolute file paths were detected in notebook code; prefer relative paths for portability.")

    return warnings


def _write_build_report(destination: Path, report: dict) -> None:
    _write_file(
        destination / "mlplant_build_report.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )


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
    extra_requirements = _detect_extra_requirements(parse_result)
    warnings = _build_warnings(parse_result, extra_requirements)

    if options.mode == "strict" and warnings:
        raise ValueError(
            "Strict mode blocked the build due to warnings: " + " | ".join(warnings)
        )

    base_context = {
        "project": options.project,
        "title": options.title,
        "port": options.port,
        "workers": options.workers,
        "mlflow": options.mlflow,
        "mlflow_tracking_uri": options.mlflow_tracking_uri,
        "docker": options.docker,
        "detected_steps": parse_result.detected_steps,
        "extra_requirements": extra_requirements,
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

    # 3b. Generate a request payload blueprint for /predict.
    request_example = _build_request_example(fields)
    _write_file(
        destination / "predict_request_example.json",
        json.dumps(request_example, indent=2, ensure_ascii=False) + "\n",
    )

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

    # 7. React UI scaffold (optional)
    if options.ui:
        _generate_react_ui(env, destination, base_context, options.notebook)

    build_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": options.mode,
        "notebook": str(Path(options.notebook).resolve()),
        "output": str(destination.resolve()),
        "detected_steps": parse_result.detected_steps,
        "extra_requirements": extra_requirements,
        "warnings": warnings,
        "flags": {
            "docker": options.docker,
            "mlflow": options.mlflow,
            "ui": options.ui,
            "ci": options.ci,
        },
    }

    options.extra["build_report"] = build_report
    if options.emit_build_report:
        _write_build_report(destination, build_report)

    fire("post_build", destination=destination, options=options)

    return destination

