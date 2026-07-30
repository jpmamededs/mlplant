<img width="1200" height="256" alt="mlplant" src="https://github.com/user-attachments/assets/5de0ae58-b28b-4337-9f85-0db5994e4319" />


mlplant is an MLOps framework that converts exploratory Jupyter notebooks into production-ready FastAPI projects using in-cell annotations.

## Installation

```bash
pip install mlplant-cli
```

## How to use

### 1. Annotate your notebook

Place an annotation at the top of each cell to mark it as a pipeline step.
Four syntax styles are supported — pick whichever fits your workflow:

#### Function call *(recommended)*

Valid before any code, reads like a registration call, no errors or warnings:

```python
import mlplant

mlplant.config()
LEARNING_RATE = 0.01
N_ESTIMATORS = 100
```

```python
mlplant.features()
X["new_feature"] = X["a"] / X["b"]
```

#### Decorator

Clean decorator syntax, but **only valid immediately before a `def` or `class`**:

```python
@mlplant.train
def _():
    model.fit(X_train, y_train)
```

```python
@mlplant.predict
def predict(data: dict):
    return model.predict([list(data.values())])[0]
```

#### Expression

Minimal form — attribute access with no call. Valid anywhere, but Pylance may
show an "expression value is unused" hint in some configurations:

```python
mlplant.load_data
df = pd.read_csv("data/train.csv")
```

#### Comment

Never causes any error or warning, useful when you want zero runtime footprint:

```python
# @mlplant.preprocessing
df = df.dropna()
df["age"] = df["age"].clip(0, 120)
```

---

All styles also accept **PascalCase aliases** (`Config`, `LoadData`, `Train`, etc.):

```python
mlplant.LoadData()        # same as mlplant.load_data()
# @Config                 # same as # @mlplant.config
@mlplant.Train            # same as @mlplant.train (before a function)
```

> Cells with the same annotation are automatically merged into a single output file.
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

## Supported annotations

| Annotation | PascalCase alias | Generated file | Purpose |
|---|---|---|---|
| `mlplant.config()` | `mlplant.Config()` | `src/config.py` | Hyperparameters and constants |
| `mlplant.load_data()` | `mlplant.LoadData()` | `src/data_loader.py` | Data ingestion |
| `mlplant.preprocessing()` | `mlplant.Preprocessing()` | `src/preprocessing.py` | Cleaning and null handling |
| `mlplant.features()` | `mlplant.Features()` | `src/features.py` | Feature engineering |
| `mlplant.train()` | `mlplant.Train()` | `src/trainer.py` | Model training |
| `mlplant.evaluate()` | `mlplant.Evaluate()` | `src/evaluate.py` | Metrics and evaluation |
| `mlplant.artifacts()` | `mlplant.Artifacts()` | `src/artifacts.py` | Model serialization |
| `mlplant.predict()` | `mlplant.Predict()` | `src/predict.py` | Inference (exposed at `/predict`) |

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
