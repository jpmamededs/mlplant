"""
decorators.py — Provides the @mlplant.<step> decorator API.

Usage inside a notebook:
    import mlplant

    @mlplant.config
    def _():
        LEARNING_RATE = 0.01

At runtime the decorators are no-ops (they return the function untouched).
The mlplant CLI parser reads the source of each cell, detects these
decorator lines and extracts the function body into production files.
"""

from __future__ import annotations
from typing import Callable


class _StepDecorator:
    """A no-op decorator that marks a cell function as a pipeline step."""

    def __init__(self, step: str) -> None:
        self.step = step

    def __call__(self, fn: Callable) -> Callable:
        return fn

    def __repr__(self) -> str:
        return f"@mlplant.{self.step}"


class _MLPlantDecorators:
    """
    Namespace that exposes all pipeline step decorators as attributes.

    After `import mlplant`, the user writes:
        @mlplant.config
        @mlplant.train
        ...etc
    """

    config = _StepDecorator("config")
    load_data = _StepDecorator("load_data")
    preprocessing = _StepDecorator("preprocessing")
    features = _StepDecorator("features")
    train = _StepDecorator("train")
    evaluate = _StepDecorator("evaluate")
    artifacts = _StepDecorator("artifacts")
    predict = _StepDecorator("predict")
