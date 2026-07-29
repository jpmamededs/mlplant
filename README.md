<img width="1200" height="256" alt="mlplant" src="https://github.com/user-attachments/assets/5de0ae58-b28b-4337-9f85-0db5994e4319" />


mlplant is an MLOps framework that converts exploratory Jupyter notebooks into production-ready FastAPI projects using in-cell annotations.

## Installation

```bash
pip install mlplant-cli
```

## How to use

### 1. Annotate your notebook

Use Python decorators directly on functions inside your notebook cells:

```python
import mlplant

@mlplant.config
def _():
    LEARNING_RATE = 0.01
    N_ESTIMATORS = 100
```

```python
@mlplant.features
def _():
    X["new_feature"] = X["a"] / X["b"]
```

```python
@mlplant.predict
def predict(data: dict):
    return model.predict([list(data.values())])[0]
```

> Cells with the same decorator are automatically merged into a single output file.
> Visual noise (matplotlib, seaborn, print, display) is stripped automatically.

### 2. Build the project

```bash
# as a direct command (after installation)
mlplant build my_notebook.ipynb --output ./api --port 8080 --title "My API" --docker

# or via python -m (works anywhere)
python -m mlplant build my_notebook.ipynb --output ./api --docker
```

### 3. Inspect detected steps

```bash
mlplant inspect my_notebook.ipynb

# or
python -m mlplant inspect my_notebook.ipynb
```

## Supported decorators

| Decorator | Generated file | Purpose |
|---|---|---|
| `@mlplant.config` | `src/config.py` | Hyperparameters and constants |
| `@mlplant.load_data` | `src/data_loader.py` | Data ingestion |
| `@mlplant.preprocessing` | `src/preprocessing.py` | Cleaning and null handling |
| `@mlplant.features` | `src/features.py` | Feature engineering |
| `@mlplant.train` | `src/trainer.py` | Model training |
| `@mlplant.evaluate` | `src/evaluate.py` | Metrics and evaluation |
| `@mlplant.artifacts` | `src/artifacts.py` | Model serialization |
| `@mlplant.predict` | `src/predict.py` | Inference function (exposed at `/predict`) |

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
