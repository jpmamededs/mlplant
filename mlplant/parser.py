"""
parser.py: Scans notebooks, identifies @mlplant.* decorators,
merges duplicate step cells, extracts function bodies, and removes visual noise.

Expected cell format:
    import mlplant

    @mlplant.features
    def _():
        X = df.drop(columns=['target'])
        X['ratio'] = X['a'] / X['b']

The decorator line and the `def` wrapper are stripped; only the body is kept.
"""

from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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

# Matches a real decorator line: @mlplant.<step>
ANNOTATION_PATTERN = re.compile(r"^@mlplant\.(\w+)\s*$", re.MULTILINE)

# Visual-only code patterns stripped during extraction.
NOISE_PATTERNS = [
    re.compile(r"^\s*(import\s+(matplotlib|seaborn|plotly)|from\s+(matplotlib|seaborn|plotly))"),
    re.compile(r"^\s*(plt\.|sns\.|fig\s*=|ax\s*=|plt\.show|plt\.savefig)"),
    re.compile(r"^\s*display\s*\("),
    re.compile(r"^\s*print\s*\("),
]


def is_noise_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in NOISE_PATTERNS)


def clean_code(code: str) -> str:
    cleaned = [line for line in code.splitlines() if not is_noise_line(line)]
    return "\n".join(cleaned).strip()


def detect_annotation(code: str) -> str | None:
    """Return the pipeline step name if the cell contains an @mlplant.<step> decorator."""
    match = ANNOTATION_PATTERN.search(code)
    if match:
        step = match.group(1)
        if step in PIPELINE_ANNOTATIONS:
            return step
    return None


def extract_body(code: str) -> str:
    """
    Given a cell with an @mlplant.<step> decorator on a function,
    return the function body dedented to column 0.

    Falls back to the full cell (minus the decorator line) when no
    wrapping function is found — this keeps backward compatibility.
    """
    lines = code.splitlines()

    # Remove the decorator line(s)
    body_start = None
    stripped_lines = []
    for i, line in enumerate(lines):
        if ANNOTATION_PATTERN.match(line.strip()):
            continue  # skip decorator
        stripped_lines.append((i, line))

    # Find the first def/async def and extract its body
    def_index = None
    for idx, (_, line) in enumerate(stripped_lines):
        if re.match(r"\s*(async\s+)?def\s+", line):
            def_index = idx
            break

    if def_index is not None:
        body_lines = [line for _, line in stripped_lines[def_index + 1:]]
        dedented = textwrap.dedent("\n".join(body_lines)).strip()
        return dedented

    # Fallback: no wrapper function — return code without the decorator line
    fallback = "\n".join(line for _, line in stripped_lines).strip()
    return fallback


class ParseResult:
    """Notebook analysis output: dictionary {step -> consolidated code}."""

    def __init__(self, blocks: Dict[str, str], metadata: Dict):
        self.blocks = blocks
        self.metadata = metadata
        self.detected_steps: List[str] = [
            step for step in PIPELINE_ANNOTATIONS if step in blocks
        ]

    def __repr__(self) -> str:
        return f"ParseResult(steps={self.detected_steps})"


def parse_notebook(notebook_path: str | Path) -> ParseResult:
    """Read notebook cells and return consolidated code grouped by pipeline step."""
    path = Path(notebook_path)
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    notebook = nbformat.read(str(path), as_version=4)
    step_blocks: Dict[str, List[str]] = defaultdict(list)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        source_code = cell.source
        step = detect_annotation(source_code)
        if step is None:
            continue

        body = extract_body(source_code)
        cleaned = clean_code(body)
        if cleaned:
            step_blocks[step].append(cleaned)

    consolidated_blocks: Dict[str, str] = {
        step: "\n\n".join(fragments) for step, fragments in step_blocks.items()
    }

    return ParseResult(blocks=consolidated_blocks, metadata=dict(notebook.metadata))
