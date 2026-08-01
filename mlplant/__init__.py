"""
mlplant — MLOps framework that converts Jupyter notebooks into production-ready API projects.

Usage in notebooks:
    import mlplant

    mlplant.config          # expression form — no syntax error, works anywhere
    mlplant.Config          # PascalCase alias

    @mlplant.config         # decorator form — only valid before a function
    def _():
        LEARNING_RATE = 0.01

    # @mlplant.config       # comment form — also recognised by the parser
"""

from mlplant.decorators import _MLPlantDecorators as _deco
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Optional
import re


def _read_version_from_pyproject() -> Optional[str]:
    """Fallback for source-tree execution when package metadata is unavailable."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        return None

    content = pyproject.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else None


def _detect_version() -> str:
    try:
        return pkg_version("mlplant-cli")
    except PackageNotFoundError:
        fallback = _read_version_from_pyproject()
        return fallback or "0.0.0"


__version__ = _detect_version()

# snake_case step markers (e.g. mlplant.config, @mlplant.train)
config = _deco.config
load_data = _deco.load_data
preprocessing = _deco.preprocessing
features = _deco.features
train = _deco.train
evaluate = _deco.evaluate
artifacts = _deco.artifacts
predict = _deco.predict

# PascalCase aliases (e.g. mlplant.Config, @mlplant.LoadData)
Config = config
LoadData = load_data
Preprocessing = preprocessing
Features = features
Train = train
Evaluate = evaluate
Artifacts = artifacts
Predict = predict
