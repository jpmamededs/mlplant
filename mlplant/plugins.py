"""plugins.py: Lightweight hook registry for the mlplant build pipeline.

Available hooks:
    pre_build   — fired before any file is generated
                  kwargs: options (BuildOptions), parse_result (ParseResult)
    post_step   — fired after each step file is written
                  kwargs: step (str), path (Path), content (str)
    post_build  — fired after all files are generated
                  kwargs: destination (Path), options (BuildOptions)

Example plugin (save as e.g. my_company/mlplant_plugin.py):

    from mlplant.plugins import register

    @register("pre_build")
    def inject_corp_logger(options, parse_result):
        ...

    @register("post_build")
    def notify_slack(destination, options):
        ...

Then register it in mlplant.yaml:
    plugins:
      - my_company.mlplant_plugin
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List

_REGISTRY: Dict[str, List[Callable]] = {}


def register(hook: str) -> Callable:
    """Decorator — register *fn* as a handler for *hook*."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY.setdefault(hook, []).append(fn)
        return fn
    return decorator


def fire(hook: str, **kwargs: Any) -> None:
    """Invoke all handlers registered for *hook* in registration order."""
    for fn in _REGISTRY.get(hook, []):
        fn(**kwargs)


def load(module_paths: List[str]) -> None:
    """Import plugin modules so they can call register() at import time."""
    for path in module_paths:
        importlib.import_module(path)
