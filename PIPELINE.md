# `pipeline.py` Functionality

`pipeline.py` defines and optionally submits an Amazon SageMaker Pipeline that trains an XGBoost regression model, evaluates it, and registers the resulting model package for downstream deployment. It is the MLOps half of this repository's end-to-end workflow: the model package it registers can later be approved and picked up by the EventBridge/CodeBuild/EKS deployment path described in `README.md`.

## High-Level Flow

The pipeline contains four ordered SageMaker steps:

| Order | Step             | Type             | Purpose                                                                                                  |
| ----- | ---------------- | ---------------- | -------------------------------------------------------------------------------------------------------- |
| 1     | `PreprocessData` | `ProcessingStep` | Reads the raw CSV from S3, prepares features and labels, and writes train/validation/test splits.        |
| 2     | `TrainXGBoost`   | `TrainingStep`   | Trains an XGBoost regression model with the SageMaker XGBoost image.                                     |
| 3     | `EvaluateModel`  | `ProcessingStep` | Downloads the trained model artifact, runs predictions on the test split, and writes evaluation metrics. |
| 4     | `RegisterModel`  | `ProcessingStep` | Creates the SageMaker Model Package Group if needed and registers the trained model package.             |

The pipeline name is `xgboost-eks-pipeline-v3-processing`.

## Processing Scripts

`pipeline.py` references three checked-in processing scripts from the local `code/` directory:

| File                     | Used by          | Responsibility                                                                                                                                                                                         |
| ------------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `code/preprocess.py`     | `PreprocessData` | Loads the source CSV from S3, validates the target column, one-hot encodes categorical columns, coerces features to numeric values, fills missing feature values with `0`, and writes split CSV files. |
| `code/evaluate.py`       | `EvaluateModel`  | Downloads `model.tar.gz` from S3, extracts `xgboost-model`, computes predictions for the test data, and writes `evaluation.json`.                                                                      |
| `code/register_model.py` | `RegisterModel`  | Ensures the model package group exists and calls `create_model_package` with model artifact, inference image, content types, and evaluation metrics.                                                   |

The pipeline resolves these files relative to `pipeline.py`, so it can find them even when the script is launched from another working directory.

## Runtime Context

The script builds a SageMaker `PipelineSession`, discovers the execution role with `get_execution_role()`, resolves the current AWS region, and uses the SageMaker default bucket for pipeline artifacts.

It retrieves two managed container images:

| Image        | Version            | Use                                                   |
| ------------ | ------------------ | ----------------------------------------------------- |
| XGBoost      | `1.7-1` / Python 3 | Model training and model evaluation.                  |
| scikit-learn | `1.2-1` / Python 3 | Preprocessing and model registration processing jobs. |

The shared S3 prefix is `xgboost-eks-pipeline`; trained models are written under `xgboost-eks-pipeline/models`.

## Pipeline Parameters

| Parameter             | Default                 | Purpose                                                                                     |
| --------------------- | ----------------------- | ------------------------------------------------------------------------------------------- |
| `InputDataS3Uri`      | none                    | Required S3 URI for the raw input CSV.                                                      |
| `TargetColumn`        | `__last__`              | Target column name. When left as `__last__`, preprocessing uses the last column in the CSV. |
| `ModelApprovalStatus` | `PendingManualApproval` | Initial approval status for the registered model package.                                   |

## Step Details

### 1. `PreprocessData`

The preprocessing job runs `code/preprocess.py` in a single `ml.m5.large` processing instance. It reads the raw CSV directly from the S3 URI passed through `InputDataS3Uri`.

Processing behavior:

- Fails if the CSV is empty.
- Uses the last column as the target when `TargetColumn` is `__last__`.
- Fails if the requested target column does not exist.
- Converts the target column to numeric values.
- Drops the target from the feature set.
- One-hot encodes categorical feature columns with `dummy_na=True`.
- Converts feature values to numeric, coercing invalid values to missing values.
- Replaces missing feature values with `0`.
- Inserts the target as the first column, which matches SageMaker's built-in XGBoost CSV expectation.
- Splits data into 70% train, 15% validation, and 15% test using `random_state=42`.

Outputs are uploaded at the end of the job:

| Output     | Local path                               | S3 prefix                                                            |
| ---------- | ---------------------------------------- | -------------------------------------------------------------------- |
| Train      | `/opt/ml/processing/train/data.csv`      | `s3://<default-bucket>/xgboost-eks-pipeline/preprocessed/train`      |
| Validation | `/opt/ml/processing/validation/data.csv` | `s3://<default-bucket>/xgboost-eks-pipeline/preprocessed/validation` |
| Test       | `/opt/ml/processing/test/data.csv`       | `s3://<default-bucket>/xgboost-eks-pipeline/preprocessed/test`       |

The split files are written without headers or indexes.

### 2. `TrainXGBoost`

The training step uses `ModelTrainer` with the SageMaker XGBoost image on one `ml.m5.xlarge` instance with a 30 GB volume.

Training inputs:

| Channel      | Source                                         |
| ------------ | ---------------------------------------------- |
| `train`      | The `PreprocessData` train output S3 URI.      |
| `validation` | The `PreprocessData` validation output S3 URI. |

Training hyperparameters:

| Hyperparameter     | Value              |
| ------------------ | ------------------ |
| `objective`        | `reg:squarederror` |
| `num_round`        | `100`              |
| `eval_metric`      | `rmse`             |
| `eta`              | `0.1`              |
| `max_depth`        | `6`                |
| `min_child_weight` | `1`                |
| `subsample`        | `0.8`              |
| `colsample_bytree` | `0.8`              |

The trained model artifact is exposed as `training_step.properties.ModelArtifacts.S3ModelArtifacts` and reused by evaluation and registration.

### 3. `EvaluateModel`

The evaluation job runs `code/evaluate.py` in the XGBoost image on one `ml.m5.large` processing instance. It receives the test split from `PreprocessData` and the trained model S3 URI from `TrainXGBoost`.

Evaluation behavior:

- Downloads `model.tar.gz` from S3.
- Extracts the artifact into `/tmp/model`.
- Loads `/tmp/model/xgboost-model` with `xgboost.Booster`.
- Reads the test split from `/opt/ml/processing/test/data.csv`.
- Treats the first column as labels and the remaining columns as features.
- Computes predictions with `xgboost.DMatrix`.
- Calculates RMSE and R2.
- Writes an `evaluation.json` report.

The report structure is:

```json
{
  "regression_metrics": {
    "rmse": { "value": 0.0 },
    "r2_score": { "value": 0.0 }
  },
  "model_s3_uri": "s3://bucket/path/model.tar.gz"
}
```

The evaluation report is declared as a SageMaker `PropertyFile`, and the script also defines `JsonGet` expressions for RMSE and R2. Those expressions are currently available for pipeline logic but are not used to gate registration.

### 4. `RegisterModel`

The registration job runs `code/register_model.py` in the scikit-learn processing image on one `ml.m5.large` processing instance.

It performs these SageMaker API operations:

1. Calls `create_model_package_group` for `xgboost-regression-models`.
2. Ignores the expected "already exists" validation error if the group is already present.
3. Calls `create_model_package` with:
   - The trained model artifact S3 URI.
   - The configured model approval status.
   - The XGBoost inference image URI.
   - Supported request and response MIME types of `text/csv`.
   - The evaluation report S3 URI as Model Quality statistics.

The registered model package is intended for the later deployment automation path, where an approved model package can trigger CodeBuild through EventBridge.

## CLI Behavior

Running the script without `--submit` only builds the pipeline object and runs preflight checks:

```bash
python3 -B ./pipeline.py
```

Submitting the pipeline requires an input CSV in S3:

```bash
python3 -B ./pipeline.py --submit \
  --input-data s3://bucket/path/data.csv \
  --target-column Rings \
  --model-approval-status PendingManualApproval
```

Add `--wait` to block until the SageMaker Pipeline execution completes:

```bash
python3 -B ./pipeline.py --submit \
  --input-data s3://bucket/path/data.csv \
  --target-column Rings \
  --model-approval-status Approved \
  --wait
```

When `--submit` is used, `submit_pipeline()`:

1. Runs local preflight checks.
2. Prints the absolute `pipeline.py` path being used.
3. Upserts the SageMaker Pipeline definition.
4. Starts a new pipeline execution with the supplied parameters.
5. Optionally waits for completion and prints the execution description.

## Preflight Checks

Two defensive checks run before pipeline submission and also when the file is executed without `--submit`:

| Check                               | What it prevents                                                                                                                                      |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `assert_no_remote_function_steps()` | Ensures the pipeline definition does not contain SageMaker SDK v3 remote-function machinery such as `FunctionStep` or remote-function bootstrap code. |
| `assert_no_stale_local_code()`      | Ensures older quality-gate or metric-argument code fragments are not present in `pipeline.py`.                                                        |

These checks help keep the implementation on ProcessingStep/TrainingStep primitives and avoid stale registration behavior.

## Important Implementation Notes

- The pipeline reads raw data from S3, not from the local `abalone.csv`.
- The preprocessing output puts the label column first because the built-in SageMaker XGBoost CSV format expects labels in the first column.
- Model registration is unconditional once evaluation completes. Metrics are recorded, but there is no active RMSE or R2 threshold.
- The model package group name is hard-coded as `xgboost-regression-models`.
- `ModelApprovalStatus` can be set to `Approved` at submission time, which can immediately activate downstream automation if the EventBridge rule is configured.
- The script intentionally avoids SageMaker SDK v3 remote-function steps, likely to avoid Python runtime mismatches between local/Studio environments and container runtimes.
