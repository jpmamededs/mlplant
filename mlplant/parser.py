"""
parser.py: Scans notebooks, identifies @mlplant.* decorators, and groups
cells into pipeline steps.

Sticky annotation rule:
    Once a cell declares @mlplant.<step>, every subsequent cell that has
    NO annotation is automatically assigned to that same step — until a
    cell with a DIFFERENT @mlplant.<step> is encountered.

    @mlplant.config   <- step = config
    def _(): ...

    X = df[...]       <- no annotation → still config
    y = df['target']  <- no annotation → still config

    @mlplant.train    <- step changes to train
    def _(): ...

Cell extraction rules:
    - `def _():` (anonymous wrapper) → body is extracted and dedented.
    - `def <name>(...):` (named function) → the full function is kept as-is.
    - No wrapper function → raw code (minus decorator line) is kept.
"""

from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nbformat

# Pipeline steps recognized by the framework, in order.
PIPELINE_ANNOTATIONS = [
    "config",
    "load_data",
    "preprocessing",
    "features",
    "train",
    "evaluate",
    "artifacts",
    "predict",
]

# PascalCase aliases → canonical step name.
PASCAL_TO_STEP: Dict[str, str] = {
    "Config": "config",
    "LoadData": "load_data",
    "Preprocessing": "preprocessing",
    "Features": "features",
    "Train": "train",
    "Evaluate": "evaluate",
    "Artifacts": "artifacts",
    "Predict": "predict",
}

_PASCAL_ALT = "|".join(PASCAL_TO_STEP)

# Supported annotation forms (each tuple: pattern, is_pascal_capture):
#   mlplant.step()  |  mlplant.step  |  @mlplant.step  |  # @mlplant.step
#   Config()        |  Config        |  @Config        |  # @Config
_ANNOT_PATTERNS: List[Tuple[re.Pattern, bool]] = [
    (re.compile(r"^@?mlplant\.(\w+)\s*\(\s*\)\s*$"), False),       # mlplant.step()  ← preferred
    (re.compile(r"^@?mlplant\.(\w+)\s*$"), False),                  # mlplant.step / @mlplant.step
    (re.compile(r"^#\s*@?mlplant\.(\w+)\s*$"), False),              # # @mlplant.step
    (re.compile(rf"^@?({_PASCAL_ALT})\s*\(\s*\)\s*$"), True),       # Config()
    (re.compile(rf"^@?({_PASCAL_ALT})\s*$"), True),                  # Config / @Config
    (re.compile(rf"^#\s*@?({_PASCAL_ALT})\s*$"), True),             # # @Config
]


def _match_annotation_line(line: str) -> Optional[str]:
    """Return canonical step name if *line* is an annotation, else None."""
    stripped = line.strip()
    for pattern, is_pascal in _ANNOT_PATTERNS:
        m = pattern.match(stripped)
        if m:
            name = m.group(1)
            step = PASCAL_TO_STEP.get(name, name) if is_pascal else name
            if step in PIPELINE_ANNOTATIONS:
                return step
    return None


def is_annotation_line(line: str) -> bool:
    return _match_annotation_line(line) is not None

# Matches an anonymous wrapper:  def _():  or  async def _():
_ANON_DEF = re.compile(r"^\s*(async\s+)?def\s+_\s*\(")

# Matches any function definition
_ANY_DEF = re.compile(r"^\s*(async\s+)?def\s+\w+\s*\(")

# Visual-only code patterns stripped during extraction.
NOISE_PATTERNS = [
    re.compile(r"^\s*(import\s+(matplotlib|seaborn|plotly)|from\s+(matplotlib|seaborn|plotly))"),
    re.compile(r"^\s*(plt\.|sns\.|fig\s*=|ax\s*=|plt\.show|plt\.savefig)"),
    re.compile(r"^\s*display\s*\("),
    re.compile(r"^\s*print\s*\("),
]


def is_noise_line(line: str) -> bool:
    return any(p.match(line) for p in NOISE_PATTERNS)


def clean_code(code: str) -> str:
    cleaned = [line for line in code.splitlines() if not is_noise_line(line)]
    return "\n".join(cleaned).strip()


def split_module_body(code: str) -> tuple:
    """Split *code* into (module_defs, body_statements).

    Top-level ``def`` / ``async def`` / ``class`` nodes are returned as
    *module_defs* — they must live at module scope, not inside a wrapper
    function.  Everything else is returned as *body_statements* and can be
    safely indented inside the template's wrapper function.

    Falls back to ("", code) when the source cannot be parsed.
    """
    if not code.strip():
        return "", ""
    try:
        import ast as _ast
        tree = _ast.parse(code)
    except SyntaxError:
        return "", code

    lines = code.splitlines()
    module_parts: List[str] = []
    body_parts: List[str] = []

    for node in tree.body:
        node_lines = lines[node.lineno - 1: node.end_lineno]
        chunk = "\n".join(node_lines)
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            module_parts.append(chunk)
        else:
            body_parts.append(chunk)

    return "\n\n".join(module_parts), "\n".join(body_parts)


def detect_annotation(code: str) -> Optional[str]:
    """Return the step name if any line in the cell is a recognised annotation."""
    for line in code.splitlines():
        step = _match_annotation_line(line)
        if step is not None:
            return step
    return None


def extract_code(source: str, has_annotation: bool) -> str:
    """
    Extract the meaningful code from a cell.

    - If the cell has an annotation and uses `def _():` as a wrapper,
      the body is extracted and dedented (the wrapper is discarded).
    - If the cell has an annotation and uses a NAMED function (e.g. `def predict(...)`),
      the full function is preserved as-is (only the decorator line is removed).
    - If the cell has no annotation (sticky step), the raw code is kept.
    """
    if not has_annotation:
        return source.strip()

    # Remove annotation lines in all supported forms
    lines_without_decorator = [
        line for line in source.splitlines()
        if not is_annotation_line(line)
    ]
    code_no_decorator = "\n".join(lines_without_decorator).strip()

    # Find the first function definition
    def_index: Optional[int] = None
    for i, line in enumerate(lines_without_decorator):
        if _ANY_DEF.match(line):
            def_index = i
            break

    if def_index is not None:
        def_line = lines_without_decorator[def_index]
        if _ANON_DEF.match(def_line):
            # Anonymous wrapper → extract and dedent the body
            body_lines = lines_without_decorator[def_index + 1:]
            return textwrap.dedent("\n".join(body_lines)).strip()
        else:
            # Named function → keep whole function
            return code_no_decorator

    # No wrapper function at all → return code minus the decorator line
    return code_no_decorator


class ParseResult:
    """Notebook analysis output: {step -> consolidated code}."""

    def __init__(self, blocks: Dict[str, str], metadata: Dict):
        self.blocks = blocks
        self.metadata = metadata
        self.detected_steps: List[str] = [
            step for step in PIPELINE_ANNOTATIONS if step in blocks
        ]

    def __repr__(self) -> str:
        return f"ParseResult(steps={self.detected_steps})"


def parse_notebook(notebook_path: str | Path) -> ParseResult:
    """
    Read notebook cells and return consolidated code grouped by pipeline step.

    Uses sticky annotation: cells without @mlplant.* are assigned to the
    last active step declared by a previous cell.
    """
    path = Path(notebook_path)
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    notebook = nbformat.read(str(path), as_version=4)
    step_blocks: Dict[str, List[str]] = defaultdict(list)
    current_step: Optional[str] = None  # sticky state

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        source = cell.source

        # Skip cells that only import mlplant itself
        if re.fullmatch(r"\s*import\s+mlplant\s*", source.strip()):
            continue

        annotation = detect_annotation(source)
        has_annotation = annotation is not None

        if has_annotation:
            current_step = annotation  # update sticky state
        elif current_step is None:
            continue  # no active step yet, skip

        active_step = current_step
        code = extract_code(source, has_annotation)
        cleaned = clean_code(code)
        if cleaned:
            step_blocks[active_step].append(cleaned)

    consolidated_blocks: Dict[str, str] = {
        step: "\n\n".join(fragments)
        for step, fragments in step_blocks.items()
    }

    return ParseResult(blocks=consolidated_blocks, metadata=dict(notebook.metadata))
