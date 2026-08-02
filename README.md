<img width="1200" height="480" alt="mlplant-bg" src="https://github.com/user-attachments/assets/0a0877e1-f6a9-4d98-858f-371a32cb2c2d" />

<hr>
mlplant is an MLOps framework that converts exploratory Jupyter notebooks into production-ready FastAPI projects using in-cell annotations.
<hr>

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

By default, `build` now also runs `train_pipeline.py` in the generated project,
so the API starts with ready-to-serve artifacts in `artifacts/`.

Data files can stay at the project root (for example `data/train.csv`).
During auto-training, mlplant keeps reading data from the notebook/project root,
but all generated pipeline outputs are written under the selected output folder.

With `--mlflow`, if a remote tracking server is unavailable, mlplant
automatically falls back to local SQLite tracking (no server required),
stored in `output/artifacts/mlflow.db`.

To generate files only (without training), use:

```bash
mlplant build my_notebook.ipynb --output ./api --no-train
```

### 3. Inspect detected steps

```bash
mlplant inspect my_notebook.ipynb

# or
python -m mlplant inspect my_notebook.ipynb
```

### 4. Run notebook diagnostics (doctor)

```bash
mlplant doctor my_notebook.ipynb

# machine-readable output
mlplant doctor my_notebook.ipynb --json

# fail CI when warnings/errors exist
mlplant doctor my_notebook.ipynb --strict

# export fix plan JSON
mlplant doctor my_notebook.ipynb --export-fixes ./doctor_fixes.json

# merge inferred optional deps into requirements
mlplant doctor my_notebook.ipynb --write-requirements ./requirements.txt

# combine strict CI with fix-plan export
mlplant doctor my_notebook.ipynb --strict --export-fixes ./doctor_fixes.json

# preview auto-fixes (dry-run by default)
mlplant doctor my_notebook.ipynb --apply

# persist auto-fixes (notebook + requirements merge when available)
mlplant doctor my_notebook.ipynb --apply --no-dry-run
```

The doctor command checks common production risks before generation, including:

- missing critical pipeline steps
- notebook runtime directives (`%...`, `!...`) inside code cells
- absolute paths that reduce portability
- third-party imports that may need explicit dependency mapping

Phase 3 doctor extras:

- exports a machine-readable fix plan (`--export-fixes`)
- can auto-merge inferred optional dependencies into a requirements file (`--write-requirements`)
- supports safe auto-fixes with dry-run preview (`--apply`, `--dry-run/--no-dry-run`)

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
  --ui            Generate React frontend scaffold for inference
  --train         Run generated train pipeline after build (default)
  --no-train      Skip training during build
  --mode          Build safety mode: flex | strict (default: flex)
  --smoke-test    Run lightweight syntax/import smoke test on generated project
  --build-report  Generate mlplant_build_report.json (default)
  --no-build-report  Skip build report generation
```

## Output layout after build

Generated code is placed in the output folder (`./output` by default):

- `output/main.py`
- `output/src/*`
- `output/train_pipeline.py`
- `output/requirements.txt`
- `output/predict_request_example.json` (payload blueprint for `/predict`)
- `output/mlplant_build_report.json` (diagnostics, warnings and inferred decisions)
- `output/Dockerfile` (when `--docker` is enabled)
- `output/ui/*` (when `--ui` is enabled)

Pipeline runtime artifacts are also kept in the output folder:

- `output/artifacts/models/*.joblib`
- `output/artifacts/metadata.json`
- `output/artifacts/label_encoder.joblib` (when available)
- `output/artifacts/mlflow.db` (local MLflow SQLite tracking)
- `output/artifacts/catboost_info/` (CatBoost auxiliary outputs)

When `--ui` is enabled:

```bash
cd output/ui
npm install
npm run dev
```

The frontend dynamically reads your API OpenAPI schema (`/openapi.json`) and
builds the predict form from the generated `InputData` contract.
