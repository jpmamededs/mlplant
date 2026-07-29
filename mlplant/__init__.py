"""
mlplant — MLOps framework that converts Jupyter notebooks into production-ready API projects.

Usage in notebooks:
    import mlplant

    @mlplant.config
    def _():
        LEARNING_RATE = 0.01
"""

from mlplant.decorators import _MLPlantDecorators as _deco

__version__ = "0.1.0"

# Expose all step decorators at the package level so users write @mlplant.config
config = _deco.config
load_data = _deco.load_data
preprocessing = _deco.preprocessing
features = _deco.features
train = _deco.train
evaluate = _deco.evaluate
artifacts = _deco.artifacts
predict = _deco.predict
