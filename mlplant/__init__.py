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

__version__ = "0.1.0"

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
