"""schema_gen.py: Infers Pydantic input fields from notebook feature engineering code."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

from mlplant.parser import ParseResult

# df['col'] = ...  or  X['col'] = ...
_COL_ASSIGN_RE = re.compile(r"""(?:df|train_df|test_df|X)\[['"](\w+)['"]\]\s*=""")

# pd.get_dummies(..., columns=['a', 'b'])
_DUMMIES_RE = re.compile(r"""pd\.get_dummies\([^)]*columns\s*=\s*\[([^\]]+)\]""")

# Patterns that imply a float result
_FLOAT_RE = re.compile(r"np\.|\.clip\(|\.abs\(|\.log|\.sqrt")


@dataclass
class FieldSpec:
    name: str
    python_type: str = "float"


def infer_fields(parse_result: ParseResult) -> List[FieldSpec]:
    """Return inferred Pydantic fields from features/preprocessing/load_data code."""
    fields: Dict[str, FieldSpec] = {}
    categoricals: Set[str] = set()

    for step in ("load_data", "features", "preprocessing"):
        code = parse_result.blocks.get(step, "")

        for m in _DUMMIES_RE.finditer(code):
            for raw in re.split(r",\s*", m.group(1)):
                col = raw.strip().strip("'\"")
                if col:
                    categoricals.add(col)

        for m in _COL_ASSIGN_RE.finditer(code):
            col = m.group(1)
            if col not in fields:
                snippet = code[m.start(): m.start() + 100]
                ptype = "float" if _FLOAT_RE.search(snippet) else "Any"
                fields[col] = FieldSpec(name=col, python_type=ptype)

    for col in categoricals:
        if col in fields:
            fields[col].python_type = "str"
        else:
            fields[col] = FieldSpec(name=col, python_type="str")

    return list(fields.values())
