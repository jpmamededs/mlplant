"""schema_gen.py: Infers Pydantic input fields from notebook feature engineering code."""
from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from typing import Dict, List, Set

from mlplant.parser import ParseResult

# df["col"]  / train_df['col'] / X["col"]
_COL_ACCESS_RE = re.compile(r"""(?:df|train_df|test_df|X)\[['"]([^'"]+)['"]\]""")

# df['col'] = ...  or  X['col'] = ...
_COL_ASSIGN_RE = re.compile(r"""^\s*(?:df|train_df|test_df|X)\[['"]([^'"]+)['"]\]\s*=""")

# pd.get_dummies(..., columns=['a', 'b'])
_DUMMIES_RE = re.compile(r"""pd\.get_dummies\([^)]*columns\s*=\s*\[([^\]]+)\]""")

# target_col = "target"
_TARGET_COL_ASSIGN_RE = re.compile(r"""\btarget_col\s*=\s*['\"]([^'\"]+)['\"]""")

# y = df['target']
_Y_FROM_DF_RE = re.compile(r"""\by\s*=\s*(?:df|train_df|test_df|X)\[['\"]([^'\"]+)['\"]\]""")

# train_df.pop('target')
_POP_TARGET_RE = re.compile(r"""(?:df|train_df|test_df|X)\.pop\(['\"]([^'\"]+)['\"]\)""")

_TYPE_PRIORITY = {
    "Any": 0,
    "bool": 1,
    "int": 2,
    "float": 3,
    "str": 4,
}


@dataclass
class FieldSpec:
    name: str
    python_name: str
    python_type: str = "float"


def _safe_python_name(raw: str) -> str:
    """Convert raw feature name into a valid Python identifier for Pydantic fields."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "feature"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_field"
    return cleaned


def _merge_type(current: str, candidate: str) -> str:
    if _TYPE_PRIORITY[candidate] > _TYPE_PRIORITY[current]:
        return candidate
    return current


def _infer_type_from_line(line: str) -> str:
    line_lower = line.lower()

    if "astype(str" in line_lower or ".str." in line_lower:
        return "str"
    if "astype(bool" in line_lower:
        return "bool"
    if "astype(int" in line_lower:
        return "int"
    if "astype(float" in line_lower:
        return "float"

    if re.search(r"==\s*['\"]|!=\s*['\"]|in\s*\[[^\]]*['\"]", line):
        return "str"
    if re.search(r"\b(is_|has_|flag|ativo|enabled|bool)\b", line_lower):
        return "bool"

    if re.search(r"\+|\-|\*|/|\.clip\(|\.abs\(|\.log|\.sqrt|np\.", line):
        return "float"
    if re.search(r"[<>]=?\s*\d+|fillna\(\s*\d+", line_lower):
        return "float"

    return "Any"


def infer_fields(parse_result: ParseResult) -> List[FieldSpec]:
    """Return inferred Pydantic fields from features/preprocessing/load_data code."""
    field_types: Dict[str, str] = {}
    categoricals: Set[str] = set()
    required_columns: Set[str] = set()
    excluded_columns: Set[str] = set()

    for step in ("load_data", "features", "preprocessing"):
        code = parse_result.blocks.get(step, "")
        if not code.strip():
            continue

        for target in _TARGET_COL_ASSIGN_RE.findall(code):
            excluded_columns.add(target)

        for target in _Y_FROM_DF_RE.findall(code):
            excluded_columns.add(target)

        for target in _POP_TARGET_RE.findall(code):
            excluded_columns.add(target)

        for m in _DUMMIES_RE.finditer(code):
            for raw in re.split(r",\s*", m.group(1)):
                col = raw.strip().strip("'\"")
                if col:
                    categoricals.add(col)
                    required_columns.add(col)

        for line in code.splitlines():
            accesses = [m.group(1) for m in _COL_ACCESS_RE.finditer(line)]
            if not accesses:
                continue

            is_assignment = _COL_ASSIGN_RE.match(line)
            if is_assignment and len(accesses) > 1:
                # lhs is generated feature, rhs accesses are required request inputs.
                read_candidates = accesses[1:]
            elif is_assignment:
                read_candidates = []
            else:
                read_candidates = accesses

            line_type = _infer_type_from_line(line)
            for col in read_candidates:
                required_columns.add(col)
                current = field_types.get(col, "Any")
                field_types[col] = _merge_type(current, line_type)

    final_columns = sorted(col for col in required_columns if col not in excluded_columns)
    used_python_names: Set[str] = set()
    fields: List[FieldSpec] = []

    for col in final_columns:
        inferred = field_types.get(col, "float")
        if col in categoricals:
            inferred = "str"
        if inferred == "Any":
            # Prefer numeric default when evidence is weak; this is a safer default
            # for tabular ML payloads than broad Any.
            inferred = "float"

        python_name = _safe_python_name(col)
        original = python_name
        suffix = 2
        while python_name in used_python_names:
            python_name = f"{original}_{suffix}"
            suffix += 1
        used_python_names.add(python_name)

        fields.append(
            FieldSpec(
                name=col,
                python_name=python_name,
                python_type=inferred,
            )
        )

    return fields
