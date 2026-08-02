"""doctor.py: Preflight notebook diagnostics for mlplant builds."""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import nbformat

from mlplant.parser import detect_annotation, parse_notebook


@dataclass
class DoctorIssue:
    severity: str
    code: str
    message: str
    hint: str


_BASE_IMPORT_ROOTS: Set[str] = {
    "mlplant",
    "fastapi",
    "uvicorn",
    "pydantic",
    "joblib",
    "pandas",
    "sklearn",
    "numpy",
    "mlflow",
}

_OPTIONAL_IMPORT_TO_PACKAGE: Dict[str, str] = {
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

_RUNTIME_RE = re.compile(r"^\s*(%{1,2}\w+|!)")
_INSTALL_RE = re.compile(r"^\s*(?:!|%{1,2})(?:pip|conda|mamba|apt|brew)\b", re.IGNORECASE)
_PLAIN_INSTALL_RE = re.compile(r"^\s*(?:pip|conda|mamba|apt|brew)\s+install\b", re.IGNORECASE)
_WINDOWS_ABS_RE = re.compile(r"[A-Za-z]:[\\/][^\n'\"]+")
_UNIX_ABS_RE = re.compile(r"(^|[\s(=])/[\w./-]+")
_CORE_STEP_ORDER = ["load_data", "train", "predict"]


def _collect_import_roots(code: str) -> Set[str]:
    roots: Set[str] = set()
    if not code.strip():
        return roots

    try:
        tree = ast.parse(code)
    except SyntaxError:
        for line in code.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m_import = re.match(r"^import\s+([A-Za-z_][\w.]*)", line)
            if m_import:
                roots.add(m_import.group(1).split(".", 1)[0])
                continue
            m_from = re.match(r"^from\s+([A-Za-z_][\w.]*)\s+import\b", line)
            if m_from:
                roots.add(m_from.group(1).split(".", 1)[0])
        return roots

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])

    return roots


def analyze_notebook(notebook_path: str | Path) -> dict:
    """Run notebook diagnostics and return a structured report."""
    path = Path(notebook_path)
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    notebook = nbformat.read(str(path), as_version=4)
    parse_result = parse_notebook(path)

    step_sequence: List[str] = []
    runtime_lines = 0
    install_lines = 0
    plain_install_lines = 0

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        source = cell.source or ""
        for line in source.splitlines():
            if _RUNTIME_RE.match(line):
                runtime_lines += 1
            if _INSTALL_RE.match(line):
                install_lines += 1
            if _PLAIN_INSTALL_RE.match(line):
                plain_install_lines += 1

        step = detect_annotation(source)
        if step is not None:
            step_sequence.append(step)

    combined_code = "\n\n".join(parse_result.blocks.values())
    import_roots = _collect_import_roots(combined_code)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        import_roots.update(_collect_import_roots(cell.source or ""))

    stdlib_roots = set(getattr(sys, "stdlib_module_names", set()))
    unknown_third_party = sorted(
        root
        for root in import_roots
        if root not in stdlib_roots
        and root not in _BASE_IMPORT_ROOTS
        and root not in _OPTIONAL_IMPORT_TO_PACKAGE
    )

    inferred_optional = sorted(
        _OPTIONAL_IMPORT_TO_PACKAGE[root]
        for root in import_roots
        if root in _OPTIONAL_IMPORT_TO_PACKAGE
    )

    issues: List[DoctorIssue] = []

    if not parse_result.detected_steps:
        issues.append(
            DoctorIssue(
                severity="error",
                code="no_steps_detected",
                message="No mlplant annotations were detected in code cells.",
                hint="Add markers like mlplant.load_data(), mlplant.train() and mlplant.predict().",
            )
        )

    if parse_result.detected_steps and "predict" not in parse_result.detected_steps:
        issues.append(
            DoctorIssue(
                severity="warning",
                code="missing_predict_step",
                message="No predict step was detected.",
                hint="Define a predict step to ensure production inference behavior is explicit.",
            )
        )

    if parse_result.detected_steps and "load_data" not in parse_result.detected_steps:
        issues.append(
            DoctorIssue(
                severity="warning",
                code="missing_load_data_step",
                message="No load_data step was detected.",
                hint="Define load_data to control training inputs and target resolution.",
            )
        )

    if runtime_lines:
        issues.append(
            DoctorIssue(
                severity="warning",
                code="runtime_directives_detected",
                message=f"Found {runtime_lines} notebook runtime directive line(s) (%/! magic or shell).",
                hint="Move package installs and shell commands outside pipeline cells when possible.",
            )
        )

    if install_lines:
        issues.append(
            DoctorIssue(
                severity="info",
                code="inline_package_install",
                message=f"Found {install_lines} inline package install directive line(s).",
                hint="Pin these packages in requirements so builds are reproducible.",
            )
        )

    if plain_install_lines:
        issues.append(
            DoctorIssue(
                severity="warning",
                code="plain_install_command_detected",
                message=f"Found {plain_install_lines} plain install command line(s) without !/% prefix.",
                hint="Use !pip or %pip in notebooks, and keep dependencies pinned in requirements.",
            )
        )

    if combined_code and (_WINDOWS_ABS_RE.search(combined_code) or _UNIX_ABS_RE.search(combined_code)):
        issues.append(
            DoctorIssue(
                severity="warning",
                code="absolute_paths_detected",
                message="Absolute paths were detected in annotated notebook code.",
                hint="Prefer relative paths to keep generated projects portable.",
            )
        )

    if unknown_third_party:
        issues.append(
            DoctorIssue(
                severity="warning",
                code="unknown_third_party_imports",
                message="Some imports may require manual dependency mapping.",
                hint="Review requirements for: " + ", ".join(unknown_third_party),
            )
        )

    severity_counts = {
        "error": sum(1 for i in issues if i.severity == "error"),
        "warning": sum(1 for i in issues if i.severity == "warning"),
        "info": sum(1 for i in issues if i.severity == "info"),
    }

    return {
        "notebook": str(path.resolve()),
        "detected_steps": parse_result.detected_steps,
        "step_sequence": step_sequence,
        "inferred_optional_dependencies": inferred_optional,
        "issues": [i.__dict__ for i in issues],
        "summary": {
            "errors": severity_counts["error"],
            "warnings": severity_counts["warning"],
            "info": severity_counts["info"],
            "total": len(issues),
        },
        "fix_plan": build_fix_plan(parse_result.detected_steps, inferred_optional, unknown_third_party),
    }


def build_fix_plan(detected_steps: List[str], inferred_optional: List[str], unknown_third_party: List[str]) -> dict:
    """Return actionable, machine-readable remediation suggestions."""
    missing_steps = [
        step for step in ("load_data", "train", "predict") if step not in detected_steps
    ]

    actions: List[dict] = []
    if missing_steps:
        actions.append(
            {
                "type": "pipeline_steps",
                "severity": "warning",
                "message": "Core pipeline annotations are missing.",
                "details": {"missing_steps": missing_steps},
            }
        )

    if inferred_optional:
        actions.append(
            {
                "type": "requirements",
                "severity": "info",
                "message": "Optional dependencies were inferred from notebook imports.",
                "details": {"dependencies": inferred_optional},
            }
        )

    if unknown_third_party:
        actions.append(
            {
                "type": "requirements_manual_review",
                "severity": "warning",
                "message": "Some third-party imports still need manual dependency mapping.",
                "details": {"import_roots": unknown_third_party},
            }
        )

    return {
        "actions": actions,
        "can_auto_fix": bool(inferred_optional),
    }


def _step_stub_source(step: str) -> str:
    if step == "load_data":
        return "\n".join(
            [
                "import mlplant",
                "",
                "mlplant.load_data()",
                "train_df = ...",
                "target_col = \"target\"",
            ]
        )

    if step == "train":
        return "\n".join(
            [
                "import mlplant",
                "",
                "mlplant.train()",
                "# Train one or more estimators here",
                "# model.fit(X_train, y_train)",
            ]
        )

    if step == "predict":
        return "\n".join(
            [
                "import mlplant",
                "",
                "mlplant.predict()",
                "def predict(data: dict):",
                "    # Return a single prediction",
                "    return 0",
            ]
        )

    return "\n".join(["import mlplant", "", f"mlplant.{step}()", "# TODO: implement this step"])


def _normalize_plain_install_lines(source: str) -> tuple[str, int]:
    lines = source.splitlines()
    changed = 0
    out: List[str] = []

    for line in lines:
        if _PLAIN_INSTALL_RE.match(line):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}!{line.lstrip()}")
            changed += 1
        else:
            out.append(line)

    return "\n".join(out), changed


def export_fix_plan(report: dict, output_path: str | Path) -> Path:
    """Write the fix plan section of a doctor report to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "notebook": report.get("notebook"),
        "summary": report.get("summary", {}),
        "fix_plan": report.get("fix_plan", {}),
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def merge_requirements(requirements_path: str | Path, dependencies: List[str]) -> dict:
    """Merge inferred dependencies into a requirements file without duplicates."""
    path = Path(requirements_path)
    existing_lines: List[str] = []
    existing_keys: Set[str] = set()

    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
        for line in existing_lines:
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            key = re.split(r"[<>=!~\[]", candidate, 1)[0].strip().lower()
            if key:
                existing_keys.add(key)

    to_add: List[str] = []
    for dep in dependencies:
        key = re.split(r"[<>=!~\[]", dep, 1)[0].strip().lower()
        if key and key not in existing_keys:
            to_add.append(dep)
            existing_keys.add(key)

    final_lines = list(existing_lines)
    if to_add:
        if final_lines and final_lines[-1].strip():
            final_lines.append("")
        final_lines.append("# Added by mlplant doctor")
        final_lines.extend(to_add)

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(final_lines).strip() + "\n" if final_lines else ""
    path.write_text(content, encoding="utf-8")

    return {
        "path": str(path.resolve()),
        "added": to_add,
        "already_present": [dep for dep in dependencies if dep not in to_add],
    }


def apply_fixes(
    notebook_path: str | Path,
    report: dict,
    dry_run: bool = True,
    requirements_path: str | Path | None = None,
) -> dict:
    """Apply safe automatic fixes. Defaults to dry-run (no writes)."""
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    notebook = nbformat.read(str(notebook_path), as_version=4)
    detected_steps = set(report.get("detected_steps", []))
    inferred_optional = list(report.get("inferred_optional_dependencies", []) or [])

    missing_core_steps = [step for step in _CORE_STEP_ORDER if step not in detected_steps]
    added_stub_cells = 0
    normalized_install_lines = 0

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        normalized, changed = _normalize_plain_install_lines(cell.source or "")
        if changed:
            normalized_install_lines += changed
            if not dry_run:
                cell.source = normalized

    if missing_core_steps and not dry_run:
        for step in missing_core_steps:
            notebook.cells.append(nbformat.v4.new_code_cell(source=_step_stub_source(step)))
            added_stub_cells += 1
    elif missing_core_steps:
        added_stub_cells = len(missing_core_steps)

    notebook_changed = normalized_install_lines > 0 or added_stub_cells > 0
    if notebook_changed and not dry_run:
        nbformat.write(notebook, str(notebook_path))

    requirements_result = None
    if requirements_path:
        if dry_run:
            requirements_result = {
                "path": str(Path(requirements_path).resolve()),
                "added": inferred_optional,
                "already_present": [],
                "dry_run": True,
            }
        else:
            requirements_result = merge_requirements(requirements_path, inferred_optional)
            requirements_result["dry_run"] = False

    return {
        "dry_run": dry_run,
        "notebook": str(notebook_path.resolve()),
        "changes": {
            "normalized_plain_install_lines": normalized_install_lines,
            "missing_core_step_stubs": missing_core_steps,
            "added_stub_cells": added_stub_cells,
            "notebook_written": bool(notebook_changed and not dry_run),
        },
        "requirements": requirements_result,
    }


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
