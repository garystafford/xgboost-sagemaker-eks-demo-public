# `pipeline.py` Functionality

`pipeline.py` defines and optionally submits an Amazon SageMaker Pipeline that prepares data, tunes an XGBoost regression model with SageMaker Automatic Model Tuning, evaluates the best candidate, records the promoted model in SageMaker managed MLflow Experiments, and registers the resulting model package for downstream deployment. It is the MLOps half of this repository's end-to-end workflow: the model package it registers can later be approved and picked up by the EventBridge/CodeBuild/EKS deployment path described in `README.md`.

## High-Level Flow

The pipeline contains five ordered SageMaker steps:

| Order | Step                         | Type             | Purpose                                                                                                                                        |
| ----- | ---------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `PreprocessData`             | `ProcessingStep` | Reads the raw CSV from S3, prepares features and labels, and writes train/validation/test splits.                                              |
| 2     | `TuneXGBoost`                | `TuningStep`     | Runs SageMaker Automatic Model Tuning and ranks candidates by validation RMSE.                                                                 |
| 3     | `EvaluateModel`              | `ProcessingStep` | Downloads the best HPO model artifact, runs predictions on the test split, and writes evaluation metrics.                                      |
| 4     | `LogPromotedModelExperiment` | `ProcessingStep` | Records the promoted model artifact, tuning job, best training job, and metrics in SageMaker managed MLflow, S3, and classic Experiments APIs. |
| 5     | `RegisterModel`              | `ProcessingStep` | Creates the SageMaker Model Package Group if needed and registers the best HPO model package.                                                  |

The pipeline name is `xgboost-eks-pipeline-v3-processing`.

## Processing Scripts

`pipeline.py` references four checked-in processing scripts from the local `code/` directory:

| File                     | Used by                      | Responsibility                                                                                                                                                                                         |
| ------------------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `code/preprocess.py`     | `PreprocessData`             | Loads the source CSV from S3, validates the target column, one-hot encodes categorical columns, coerces features to numeric values, fills missing feature values with `0`, and writes split CSV files. |
| `code/evaluate.py`       | `EvaluateModel`              | Downloads `model.tar.gz` from S3, extracts `xgboost-model`, computes predictions for the test data, and writes `evaluation.json`.                                                                      |
| `code/log_experiment.py` | `LogPromotedModelExperiment` | Logs the selected HPO model to SageMaker managed MLflow when configured, writes a summary artifact to S3, and creates or updates classic Experiment/Trial/Trial Component records.                     |
| `code/register_model.py` | `RegisterModel`              | Ensures the model package group exists and calls `create_model_package` with model artifact, inference image, content types, and evaluation metrics.                                                   |

The pipeline resolves these files relative to `pipeline.py`, so it can find them even when the script is launched from another working directory.

## Runtime Context

The script builds a SageMaker `PipelineSession`, discovers the execution role with `get_execution_role()`, resolves the current AWS region, and uses the SageMaker default bucket for pipeline artifacts.

It retrieves two managed container images:

| Image        | Version            | Use                                                                   |
| ------------ | ------------------ | --------------------------------------------------------------------- |
| XGBoost      | `1.7-1` / Python 3 | Model tuning candidate jobs and model evaluation.                     |
| scikit-learn | `1.2-1` / Python 3 | Preprocessing, Experiments logging, and registration processing jobs. |

The shared S3 prefix is `xgboost-eks-pipeline`; trained models are written under `xgboost-eks-pipeline/models`.

## Pipeline Parameters

| Parameter                 | Default                   | Purpose                                                                                                                        |
| ------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `InputDataS3Uri`          | none                      | Required S3 URI for the raw input CSV.                                                                                         |
| `TargetColumn`            | `__last__`                | Target column name. When left as `__last__`, preprocessing uses the last column in the CSV.                                    |
| `ModelApprovalStatus`     | `PendingManualApproval`   | Initial approval status for the registered model package.                                                                      |
| `ExperimentName`          | `xgboost-eks-experiments` | Experiment name used for pipeline executions and MLflow runs.                                                                  |
| `MlflowTrackingServerArn` | `__disabled__`            | Optional SageMaker managed MLflow tracking server ARN. When disabled, only S3 and classic Experiments API records are written. |

The HPO job counts are definition-time CLI options, not SageMaker Pipeline runtime parameters. `--max-tuning-jobs` and `--max-parallel-tuning-jobs` update the pipeline definition before `pipeline.upsert(...)`.

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

### 2. `TuneXGBoost`

The tuning step uses `HyperparameterTuner` with a `ModelTrainer` configured for the SageMaker XGBoost image. SageMaker Automatic Model Tuning launches up to the CLI-configured candidate training job limit, with up to the CLI-configured parallel job limit running at once. Each candidate uses one `ml.m5.xlarge` instance with a 30 GB volume.

Tuning inputs:

| Channel      | Source                                         |
| ------------ | ---------------------------------------------- |
| `train`      | The `PreprocessData` train output S3 URI.      |
| `validation` | The `PreprocessData` validation output S3 URI. |

Static hyperparameters:

| Hyperparameter     | Value              |
| ------------------ | ------------------ |
| `objective`        | `reg:squarederror` |
| `num_round`        | `100`              |
| `eval_metric`      | `rmse`             |
| `eta`              | Tuned              |
| `max_depth`        | Tuned              |
| `min_child_weight` | Tuned              |
| `subsample`        | Tuned              |
| `colsample_bytree` | Tuned              |

Tuned hyperparameter ranges:

| Hyperparameter     | Range           | Type                    |
| ------------------ | --------------- | ----------------------- |
| `eta`              | `0.01` to `0.3` | Continuous, logarithmic |
| `max_depth`        | `3` to `10`     | Integer                 |
| `min_child_weight` | `1` to `10`     | Continuous              |
| `subsample`        | `0.5` to `1.0`  | Continuous              |
| `colsample_bytree` | `0.5` to `1.0`  | Continuous              |

The tuning objective is `validation:rmse`, minimized with Bayesian search and automatic early stopping. The best model artifact is resolved through `tuning_step.get_top_model_s3_uri(top_k=0, ...)` and reused by evaluation, Experiments logging, and registration.

### 3. `EvaluateModel`

The evaluation job runs `code/evaluate.py` in the XGBoost image on one `ml.m5.large` processing instance. It receives the test split from `PreprocessData` and the best model S3 URI from `TuneXGBoost`.

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

### 4. `LogPromotedModelExperiment`

The Experiments logging job runs `code/log_experiment.py` in the scikit-learn processing image on one `ml.m5.large` processing instance.

When `MlflowTrackingServerArn` is set, it writes a run to the configured SageMaker managed MLflow tracking server. The run uses the configured `ExperimentName`, the SageMaker pipeline execution ID as the run name, and records:

- RMSE and R2 as metrics.
- The model package group, tuning job, and best training job as parameters.
- The best model artifact S3 URI and evaluation report S3 URI as tags.
- The local evaluation report and experiment summary as best-effort MLflow artifacts.

The script installs `mlflow` and `sagemaker-mlflow` before importing `boto3` if those packages are missing from the processing image. That avoids mixing an already-imported AWS SDK module with packages upgraded by pip in the same Python process. Artifact upload is best-effort; metrics, parameters, tags, and the S3 summary are still recorded if MLflow artifact upload fails.

The current SageMaker Studio Experiments page is MLflow-backed, so this is the path that makes the run visible in Studio. If `MlflowTrackingServerArn` is left as `__disabled__`, the step still writes S3 and classic SageMaker Experiments API records, but those records may not appear in the current Studio UI.

For compatibility, it also performs these classic SageMaker API operations:

1. Ensures the configured SageMaker Experiment exists.
2. Ensures the pipeline execution Trial exists.
3. Creates or updates a promoted-model Trial Component.
4. Associates that Trial Component with the pipeline execution Trial.

The Trial Component records:

- The best model artifact S3 URI.
- The evaluation report S3 URI.
- The tuning job name.
- The best training job name.
- RMSE and R2 as numeric parameters.
- The model package group name.

It always writes `experiment_summary.json` under `s3://<default-bucket>/xgboost-eks-pipeline/experiments/<trial-name>/`.

### 5. `RegisterModel`

The registration job runs `code/register_model.py` in the scikit-learn processing image on one `ml.m5.large` processing instance.

It performs these SageMaker API operations:

1. Calls `create_model_package_group` for `xgboost-regression-models`.
2. Ignores the expected "already exists" validation error if the group is already present.
3. Calls `create_model_package` with:
   - The best HPO model artifact S3 URI.
   - The configured model approval status.
   - The XGBoost inference image URI.
   - Supported request and response MIME types of `text/csv`.
   - The evaluation report S3 URI as Model Quality statistics.

The registered model package is intended for the later deployment automation path, where an approved model package can trigger CodeBuild through EventBridge. The EventBridge target passes the approved model package ARN into CodeBuild as `MODEL_PACKAGE_ARN`, so the deployment job uses the exact package that triggered the event.

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
  --model-approval-status PendingManualApproval \
  --experiment-name xgboost-eks-experiments \
  --mlflow-tracking-server-name ml-flow-tracking-demo \
  --max-tuning-jobs 8 \
  --max-parallel-tuning-jobs 2
```

Add `--wait` to block until the SageMaker Pipeline execution completes:

```bash
python3 -B ./pipeline.py --submit \
  --input-data s3://bucket/path/data.csv \
  --target-column Rings \
  --model-approval-status Approved \
  --experiment-name xgboost-eks-experiments \
  --mlflow-tracking-server-name ml-flow-tracking-demo \
  --max-tuning-jobs 8 \
  --max-parallel-tuning-jobs 2 \
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

These checks help keep the implementation on ProcessingStep/TuningStep primitives and avoid stale registration behavior.

## Important Implementation Notes

- The pipeline reads raw data from S3, not from the local `abalone.csv`.
- The preprocessing output puts the label column first because the built-in SageMaker XGBoost CSV format expects labels in the first column.
- SageMaker Automatic Model Tuning chooses the promoted model artifact. There is no separate final retraining step after HPO.
- Pipeline executions are grouped in MLflow using `ExperimentName` and the SageMaker pipeline execution ID as the run name. Classic Experiments API objects use the same experiment and trial names for compatibility.
- Model registration is unconditional once evaluation completes. Metrics are recorded, but there is no active RMSE or R2 threshold.
- The model package group name is hard-coded as `xgboost-regression-models`.
- `ModelApprovalStatus` can be set to `Approved` at submission time, which can immediately activate downstream automation if the EventBridge rule is configured.
- The script intentionally avoids SageMaker SDK v3 remote-function steps, likely to avoid Python runtime mismatches between local/Studio environments and container runtimes.
