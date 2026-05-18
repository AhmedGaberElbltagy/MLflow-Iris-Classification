# MLflow Iris Classification

This project demonstrates a small end-to-end MLflow workflow for training, registering, promoting, and using a machine learning model.

The example trains Random Forest classifiers on the Iris dataset, logs each run to MLflow, registers the model, promotes the best model version to a `production` alias, and loads that alias for prediction.

## Project Structure

```text
.
├── train.py            # Trains and logs Iris classifiers to MLflow
├── promote_model.py    # Finds the best registered model version and promotes it
├── predict.py          # Loads the production model alias and runs a sample prediction
├── requirements.txt    # Python dependencies
├── mlflow.db           # Local MLflow backend database
└── mlruns/             # Local MLflow artifacts and run data
```

## Requirements

- Python 3.9+
- MLflow
- scikit-learn
- pandas
- numpy

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Start MLflow

The scripts use this tracking URI:

```text
http://127.0.0.1:5000
```

Start the MLflow tracking server before running the scripts:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Open the MLflow UI in your browser:

```text
http://127.0.0.1:5000
```

## Train Models

Run:

```bash
python train.py
```

This script:

- Loads the Iris dataset
- Splits it into train and test sets
- Trains three `RandomForestClassifier` models with different hyperparameters
- Logs parameters and accuracy to MLflow
- Registers each trained model as `iris-random-forest`

## Promote the Best Model

After training, promote the registered model version with the highest logged accuracy:

```bash
python promote_model.py
```

This script:

- Searches all versions of the `iris-random-forest` registered model
- Reads each version's logged `accuracy`
- Selects the best version
- Assigns it the MLflow alias `production`
- Adds a `stage=production` tag

## Run Prediction

Once a model version has been promoted, run:

```bash
python predict.py
```

The prediction script loads:

```text
models:/iris-random-forest@production
```

and predicts the class for this sample Iris measurement:

```python
[[5.1, 3.5, 1.4, 0.2]]
```

## Typical Workflow

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
python train.py
python promote_model.py
python predict.py
```

Keep the MLflow server running while executing the Python scripts.

## Notes

- The project uses a local SQLite database, `mlflow.db`, for MLflow tracking metadata.
- Model artifacts are stored locally in the `mlruns/` directory.
- `predict.py` requires a registered model version with the `production` alias. Run `train.py` and `promote_model.py` first if the alias does not exist.
