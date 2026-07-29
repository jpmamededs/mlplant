# mlplant

mlplant is an MLOps framework that converts exploratory Jupyter notebooks into production-ready FastAPI projects using in-cell annotations.

## Installation

```bash
pip install -e .
```

## How to use

### 1. Annotate your notebook

Add structured comments at the top of your code cells:

```python
# @mlplant.config
LEARNING_RATE = 0.01
N_ESTIMATORS = 100
```

```python
# @mlplant.features
X["new_feature"] = X["a"] / X["b"]
```

```python
# @mlplant.predict
def predict(data: dict):
    return model.predict([list(data.values())])[0]
```

### 2. Build the project

```bash
mlplant build my_notebook.ipynb --output ./api --port 8080 --title "My API" --docker
```

### 3. Inspect detected annotations

```bash
mlplant inspect my_notebook.ipynb
```

## Supported annotations

| Annotation | Generated file |
|---|---|
| `# @mlplant.config` | `src/config.py` |
| `# @mlplant.load_data` | `src/data_loader.py` |
| `# @mlplant.preprocessing` | `src/preprocessing.py` |
| `# @mlplant.features` | `src/features.py` |
| `# @mlplant.train` | `src/trainer.py` |
| `# @mlplant.evaluate` | `src/evaluate.py` |
| `# @mlplant.artifacts` | `src/artifacts.py` |
| `# @mlplant.predict` | `src/predict.py` |

## Build flags

```
mlplant build <notebook> [OPTIONS]

  --output / -o   Output directory             (default: ./output)
  --port          Server port                  (default: 8000)
  --title         Swagger UI title             (default: "mlplant API")
  --workers       Uvicorn workers              (default: 1)
  --docker        Generate optimized Dockerfile
  --mlflow        Inject MLflow tracking into training
```
