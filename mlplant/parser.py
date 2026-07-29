"""
parser.py: Scans notebooks, identifies @mlplant.* annotations,
merges duplicate step cells, and removes visual-noise lines.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import nbformat

# Annotations recognized by the framework, in pipeline order.
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

# Regex pattern to capture: # @mlplant.<step>
ANNOTATION_PATTERN = re.compile(r"#\s*@mlplant\.(\w+)")

# Visual-only code patterns ignored during extraction.
NOISE_PATTERNS = [
    re.compile(r"^\s*(import\s+(matplotlib|seaborn|plotly)|from\s+(matplotlib|seaborn|plotly))"),
    re.compile(r"^\s*(plt\.|sns\.|fig\s*=|ax\s*=|plt\.show|plt\.savefig)"),
    re.compile(r"^\s*display\s*\("),
    re.compile(r"^\s*print\s*\("),
]


def is_noise_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in NOISE_PATTERNS)


def clean_code(code: str) -> str:
    cleaned_lines = [line for line in code.splitlines() if not is_noise_line(line)]
    return "\n".join(cleaned_lines).strip()


def detect_annotation(code: str) -> str | None:
    for line in code.splitlines():
        match = ANNOTATION_PATTERN.search(line)
        if match:
            step = match.group(1)
            if step in PIPELINE_ANNOTATIONS:
                return step
    return None


def remove_annotation_line(code: str) -> str:
    lines = [line for line in code.splitlines() if not ANNOTATION_PATTERN.search(line)]
    return "\n".join(lines).strip()


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

        code_without_annotation = remove_annotation_line(source_code)
        cleaned_code = clean_code(code_without_annotation)
        if cleaned_code:
            step_blocks[step].append(cleaned_code)

    consolidated_blocks: Dict[str, str] = {
        step: "\n\n".join(fragments) for step, fragments in step_blocks.items()
    }

    return ParseResult(blocks=consolidated_blocks, metadata=dict(notebook.metadata))
