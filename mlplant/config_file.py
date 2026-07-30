"""config_file.py: Loads mlplant.yaml project configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

_CONFIG_NAMES = ("mlplant.yaml", "mlplant.yml")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def find_and_load(search_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Look for mlplant.yaml starting from *search_dir* (or cwd) and return its contents."""
    base = Path(search_dir) if search_dir else Path.cwd()
    for name in _CONFIG_NAMES:
        candidate = base / name
        if candidate.exists():
            return _load_yaml(candidate)
    return {}
