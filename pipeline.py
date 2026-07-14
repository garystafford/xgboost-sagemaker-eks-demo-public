import argparse
from pathlib import Path

try:
    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import get_execution_role
    from sagemaker.core.parameter import ContinuousParameter, IntegerParameter
    from sagemaker.core.processing import ScriptProcessor
    from sagemaker.core.shapes import (
        Channel,
        DataSource,
        ProcessingInput,
        ProcessingOutput,
        ProcessingS3Input,
        ProcessingS3Output,
        S3DataSource,
    )
    from sagemaker.core.training.configs import Compute, OutputDataConfig
    from sagemaker.core.workflow.execution_variables import ExecutionVariables
    from sagemaker.core.workflow.functions import JsonGet, Join
    from sagemaker.core.workflow.parameters import ParameterString
    from sagemaker.core.workflow.pipeline_context import PipelineSession
    from sagemaker.core.workflow.properties import PropertyFile
    from sagemaker.mlops.workflow.pipeline import Pipeline
    from sagemaker.mlops.workflow.pipeline_experiment_config import (
        PipelineExperimentConfig,
        PipelineExperimentConfigProperties,
    )
    from sagemaker.mlops.workflow.steps import ProcessingStep, TuningStep
    from sagemaker.train import ModelTrainer
    from sagemaker.train.tuner import HyperparameterTuner
except ModuleNotFoundError as exc:
    if exc.name != "sagemaker":
        raise
    raise SystemExit(
        "Missing SageMaker Python SDK. Install the project dependencies with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        "Then rerun pipeline.py from SageMaker Studio or another environment "
        "where get_execution_role() can resolve a SageMaker execution role."
    ) from exc


CODE_DIR = Path(__file__).resolve().parent / "code"

pipeline_session = PipelineSession()
role = get_execution_role()
region = pipeline_session.boto_region_name
bucket = pipeline_session.default_bucket()

pipeline_prefix = "xgboost-eks-pipeline"
model_prefix = f"{pipeline_prefix}/models"
DEFAULT_MAX_TUNING_JOBS = 8
DEFAULT_MAX_PARALLEL_TUNING_JOBS = 2

xgboost_image = image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1",
    py_version="py3",
    instance_type="ml.m5.xlarge",
)

sklearn_processing_image = image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large",
)

clarify_image = image_uris.retrieve(
    framework="clarify",
    region=region,
    version="1.0",
)

input_data_uri = ParameterString(name="InputDataS3Uri")
target_column = ParameterString(name="TargetColumn", default_value="__last__")
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
)
experiment_name = ParameterString(
    name="ExperimentName", default_value="xgboost-eks-experiments"
)
mlflow_tracking_server_arn = ParameterString(
    name="MlflowTrackingServerArn", default_value="__disabled__"
)


# --- Step 1: Preprocess with ProcessingStep ---
preprocess_processor = ScriptProcessor(
    image_uri=sklearn_processing_image,
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name=f"{pipeline_prefix}-preprocess",
    sagemaker_session=pipeline_session,
)

preprocess_args = preprocess_processor.run(
    code=str(CODE_DIR / "preprocess.py"),
    arguments=["--input-data", input_data_uri, "--target-column", target_column],
    outputs=[
        ProcessingOutput(
            output_name="train",
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/{pipeline_prefix}/preprocessed/train",
                local_path="/opt/ml/processing/train",
                s3_upload_mode="EndOfJob",
            ),
        ),
        ProcessingOutput(
            output_name="validation",
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/{pipeline_prefix}/preprocessed/validation",
                local_path="/opt/ml/processing/validation",
                s3_upload_mode="EndOfJob",
            ),
        ),
        ProcessingOutput(
            output_name="test",
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/{pipeline_prefix}/preprocessed/test",
                local_path="/opt/ml/processing/test",
                s3_upload_mode="EndOfJob",
            ),
        ),
        ProcessingOutput(
            output_name="analysis",
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/{pipeline_prefix}/preprocessed/analysis",
                local_path="/opt/ml/processing/analysis",
                s3_upload_mode="EndOfJob",
            ),
        ),
    ],
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    step_args=preprocess_args,
)


# --- Step 2: Tune XGBoost with SageMaker Automatic Model Tuning ---
xgboost_trainer = ModelTrainer(
    training_image=xgboost_image,
    role=role,
    sagemaker_session=pipeline_session,
    base_job_name="xgboost-eks-tune",
    compute=Compute(
        instance_type="ml.m5.xlarge",
        instance_count=1,
        volume_size_in_gb=30,
    ),
    output_data_config=OutputDataConfig(
        s3_output_path=f"s3://{bucket}/{model_prefix}/"
    ),
    hyperparameters={
        "objective": "reg:squarederror",
        "num_round": "100",
        "eval_metric": "rmse",
        "eta": "0.1",
        "max_depth": "6",
        "min_child_weight": "1",
        "subsample": "0.8",
        "colsample_bytree": "0.8",
    },
)

tuner = HyperparameterTuner(
    model_trainer=xgboost_trainer,
    objective_metric_name="validation:rmse",
    hyperparameter_ranges={
        "eta": ContinuousParameter(0.01, 0.3, scaling_type="Logarithmic"),
        "max_depth": IntegerParameter(3, 10),
        "min_child_weight": ContinuousParameter(1, 10),
        "subsample": ContinuousParameter(0.5, 1.0),
        "colsample_bytree": ContinuousParameter(0.5, 1.0),
    },
    objective_type="Minimize",
    strategy="Bayesian",
    metric_definitions=[
        {
            "Name": "validation:rmse",
            "Regex": r".*validation-rmse:([-+0-9.eE]+).*",
        }
    ],
    max_jobs=DEFAULT_MAX_TUNING_JOBS,
    max_parallel_jobs=DEFAULT_MAX_PARALLEL_TUNING_JOBS,
    early_stopping_type="Auto",
    base_tuning_job_name="xgboost-eks-hpo",
)

tuning_args = tuner.tune(
    inputs=[
        Channel(
            channel_name="train",
            data_source=DataSource(
                s3_data_source=S3DataSource(
                    s3_data_type="S3Prefix",
                    s3_uri=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                        "train"
                    ].S3Output.S3Uri,
                    s3_data_distribution_type="FullyReplicated",
                )
            ),
            content_type="text/csv",
        ),
        Channel(
            channel_name="validation",
            data_source=DataSource(
                s3_data_source=S3DataSource(
                    s3_data_type="S3Prefix",
                    s3_uri=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                        "validation"
                    ].S3Output.S3Uri,
                    s3_data_distribution_type="FullyReplicated",
                )
            ),
            content_type="text/csv",
        ),
    ],
    wait=False,
)

tuning_step = TuningStep(
    name="TuneXGBoost",
    step_args=tuning_args,
    depends_on=[preprocess_step],
)

best_model_s3_uri = tuning_step.get_top_model_s3_uri(
    top_k=0,
    s3_bucket=bucket,
    prefix=model_prefix,
)


# --- Step 3: Evaluate with ProcessingStep ---
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

evaluation_processor = ScriptProcessor(
    image_uri=xgboost_image,
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name=f"{pipeline_prefix}-evaluate",
    sagemaker_session=pipeline_session,
)

evaluation_args = evaluation_processor.run(
    code=str(CODE_DIR / "evaluate.py"),
    inputs=[
        ProcessingInput(
            input_name="test",
            s3_input=ProcessingS3Input(
                s3_uri=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                local_path="/opt/ml/processing/test",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/{pipeline_prefix}/evaluation",
                local_path="/opt/ml/processing/evaluation",
                s3_upload_mode="EndOfJob",
            ),
        )
    ],
    arguments=[
        "--model-s3-uri",
        best_model_s3_uri,
        "--test-data-dir",
        "/opt/ml/processing/test",
        "--output-dir",
        "/opt/ml/processing/evaluation",
    ],
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    step_args=evaluation_args,
    property_files=[evaluation_report],
    depends_on=[tuning_step],
)

evaluation_s3_uri = Join(
    on="/",
    values=[
        evaluation_step.properties.ProcessingOutputConfig.Outputs[
            "evaluation"
        ].S3Output.S3Uri,
        "evaluation.json",
    ],
)

r2_score = JsonGet(
    step_name=evaluation_step.name,
    property_file=evaluation_report,
    json_path="regression_metrics.r2_score.value",
)

rmse = JsonGet(
    step_name=evaluation_step.name,
    property_file=evaluation_report,
    json_path="regression_metrics.rmse.value",
)


# --- Step 4: Run SageMaker Clarify SHAP explainability ---
clarify_output_s3_uri = Join(
    on="/",
    values=[
        f"s3://{bucket}/{pipeline_prefix}/clarify",
        PipelineExperimentConfigProperties.TRIAL_NAME,
    ],
)

clarify_report_s3_uri = Join(
    on="/",
    values=[
        clarify_output_s3_uri,
        "analysis",
        "analysis.json",
    ],
)

analysis_data_s3_uri = Join(
    on="/",
    values=[
        preprocess_step.properties.ProcessingOutputConfig.Outputs[
            "analysis"
        ].S3Output.S3Uri,
        "data.csv",
    ],
)

clarify_launcher = ScriptProcessor(
    image_uri=sklearn_processing_image,
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name=f"{pipeline_prefix}-clarify-launcher",
    sagemaker_session=pipeline_session,
)

clarify_args = clarify_launcher.run(
    code=str(CODE_DIR / "run_clarify.py"),
    inputs=[
        ProcessingInput(
            input_name="analysis",
            s3_input=ProcessingS3Input(
                s3_uri=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "analysis"
                ].S3Output.S3Uri,
                local_path="/opt/ml/processing/analysis",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="clarify_summary",
            s3_output=ProcessingS3Output(
                s3_uri=Join(on="/", values=[clarify_output_s3_uri, "summary"]),
                local_path="/opt/ml/processing/clarify_summary",
                s3_upload_mode="EndOfJob",
            ),
        )
    ],
    arguments=[
        "--analysis-data-dir",
        "/opt/ml/processing/analysis",
        "--analysis-data-s3-uri",
        analysis_data_s3_uri,
        "--model-s3-uri",
        best_model_s3_uri,
        "--inference-image-uri",
        xgboost_image,
        "--clarify-image-uri",
        clarify_image,
        "--clarify-output-s3-uri",
        clarify_output_s3_uri,
        "--summary-output-dir",
        "/opt/ml/processing/clarify_summary",
        "--role-arn",
        role,
        "--region",
        region,
        "--model-name",
        Join(
            on="-",
            values=[
                "clarify",
                "xgb",
                "model",
                PipelineExperimentConfigProperties.TRIAL_NAME,
            ],
        ),
        "--clarify-job-name",
        Join(
            on="-",
            values=[
                "clarify",
                "xgb",
                PipelineExperimentConfigProperties.TRIAL_NAME,
            ],
        ),
    ],
)

clarify_step = ProcessingStep(
    name="RunClarifyExplainability",
    step_args=clarify_args,
    depends_on=[evaluation_step],
)


# --- Step 5: Log the promoted model to SageMaker Experiments ---
experiment_output_s3_uri = Join(
    on="/",
    values=[
        f"s3://{bucket}/{pipeline_prefix}/experiments",
        PipelineExperimentConfigProperties.TRIAL_NAME,
    ],
)

experiment_report_s3_uri = Join(
    on="/",
    values=[
        experiment_output_s3_uri,
        "experiment_summary.json",
    ],
)

experiment_logger = ScriptProcessor(
    image_uri=sklearn_processing_image,
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name=f"{pipeline_prefix}-experiment",
    sagemaker_session=pipeline_session,
)

experiment_args = experiment_logger.run(
    code=str(CODE_DIR / "log_experiment.py"),
    inputs=[
        ProcessingInput(
            input_name="evaluation",
            s3_input=ProcessingS3Input(
                s3_uri=evaluation_step.properties.ProcessingOutputConfig.Outputs[
                    "evaluation"
                ].S3Output.S3Uri,
                local_path="/opt/ml/processing/evaluation",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="experiment",
            s3_output=ProcessingS3Output(
                s3_uri=experiment_output_s3_uri,
                local_path="/opt/ml/processing/experiment",
                s3_upload_mode="EndOfJob",
            ),
        )
    ],
    arguments=[
        "--experiment-name",
        PipelineExperimentConfigProperties.EXPERIMENT_NAME,
        "--mlflow-tracking-server-arn",
        mlflow_tracking_server_arn,
        "--trial-name",
        PipelineExperimentConfigProperties.TRIAL_NAME,
        "--trial-component-display-name",
        "PromotedXGBoostModel",
        "--model-s3-uri",
        best_model_s3_uri,
        "--evaluation-s3-uri",
        evaluation_s3_uri,
        "--clarify-report-s3-uri",
        clarify_report_s3_uri,
        "--evaluation-report-path",
        "/opt/ml/processing/evaluation/evaluation.json",
        "--summary-output-dir",
        "/opt/ml/processing/experiment",
        "--summary-s3-uri",
        experiment_report_s3_uri,
        "--model-package-group-name",
        "xgboost-regression-models",
        "--tuning-job-name",
        tuning_step.properties.HyperParameterTuningJobName,
        "--best-training-job-name",
        tuning_step.properties.TrainingJobSummaries[0].TrainingJobName,
        "--region",
        region,
    ],
)

experiment_step = ProcessingStep(
    name="LogPromotedModelExperiment",
    step_args=experiment_args,
    depends_on=[clarify_step],
)


# --- Step 6: Register with a ProcessingStep ---
register_processor = ScriptProcessor(
    image_uri=sklearn_processing_image,
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name=f"{pipeline_prefix}-register",
    sagemaker_session=pipeline_session,
)

register_args = register_processor.run(
    code=str(CODE_DIR / "register_model.py"),
    arguments=[
        "--model-s3-uri",
        best_model_s3_uri,
        "--metrics-s3-uri",
        evaluation_s3_uri,
        "--explainability-report-s3-uri",
        clarify_report_s3_uri,
        "--approval-status",
        model_approval_status,
        "--inference-image-uri",
        xgboost_image,
        "--model-package-group-name",
        "xgboost-regression-models",
        "--region",
        region,
    ],
)

register_step = ProcessingStep(
    name="RegisterModel",
    step_args=register_args,
    depends_on=[experiment_step],
)

pipeline = Pipeline(
    name="xgboost-eks-pipeline-v3-processing",
    parameters=[
        input_data_uri,
        target_column,
        model_approval_status,
        experiment_name,
        mlflow_tracking_server_arn,
    ],
    pipeline_experiment_config=PipelineExperimentConfig(
        experiment_name=experiment_name,
        trial_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
    ),
    steps=[
        preprocess_step,
        tuning_step,
        evaluation_step,
        clarify_step,
        experiment_step,
        register_step,
    ],
    sagemaker_session=pipeline_session,
)


def assert_no_remote_function_steps():
    definition = pipeline.definition()
    forbidden_terms = [
        "sagemaker_remote_function_bootstrap",
        "remote_function",
        "FunctionStep",
        "pipeline.preprocess",
        "client_python_version",
    ]
    found = [term for term in forbidden_terms if term in definition]
    if found:
        raise RuntimeError(
            "Pipeline definition still contains SDK v3 remote-function machinery: "
            + ", ".join(found)
        )
    print(
        "Preflight passed: pipeline definition contains no SDK v3 remote-function steps."
    )


def assert_no_stale_local_code():
    source_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_terms = [
        "Check" + "ModelQuality",
        "Condition" + "Step",
        "--" + "rmse",
        "--" + "r2-score",
        "Customer" + "MetadataProperties",
        "min" + "-r2-score",
    ]
    found = [term for term in forbidden_terms if term in source_text]
    if found:
        raise RuntimeError(
            "This pipeline.py still contains stale code that will break registration: "
            + ", ".join(found)
        )
    print("Preflight passed: no stale quality-gate or metric-argument code found.")


def configure_tuning_limits(max_tuning_jobs_value, max_parallel_tuning_jobs_value):
    if max_tuning_jobs_value < 1:
        raise ValueError("--max-tuning-jobs must be at least 1.")
    if max_parallel_tuning_jobs_value < 1:
        raise ValueError("--max-parallel-tuning-jobs must be at least 1.")
    if max_parallel_tuning_jobs_value > max_tuning_jobs_value:
        raise ValueError("--max-parallel-tuning-jobs cannot exceed --max-tuning-jobs.")

    tuner.max_jobs = max_tuning_jobs_value
    tuner.max_parallel_jobs = max_parallel_tuning_jobs_value


def resolve_mlflow_tracking_server_arn(
    mlflow_tracking_server_arn_value, mlflow_tracking_server_name_value
):
    if mlflow_tracking_server_arn_value and mlflow_tracking_server_name_value:
        raise ValueError(
            "Use either --mlflow-tracking-server-arn or "
            "--mlflow-tracking-server-name, not both."
        )

    if mlflow_tracking_server_arn_value:
        return mlflow_tracking_server_arn_value

    if not mlflow_tracking_server_name_value:
        return "__disabled__"

    import boto3

    sm_client = boto3.client("sagemaker", region_name=region)
    response = sm_client.describe_mlflow_tracking_server(
        TrackingServerName=mlflow_tracking_server_name_value
    )
    return response["TrackingServerArn"]


def submit_pipeline(
    input_data_s3_uri,
    target_column_name="__last__",
    model_approval_status_value="PendingManualApproval",
    experiment_name_value="xgboost-eks-experiments",
    mlflow_tracking_server_arn_value="",
    mlflow_tracking_server_name_value="",
    max_tuning_jobs_value=DEFAULT_MAX_TUNING_JOBS,
    max_parallel_tuning_jobs_value=DEFAULT_MAX_PARALLEL_TUNING_JOBS,
    wait=False,
):
    configure_tuning_limits(max_tuning_jobs_value, max_parallel_tuning_jobs_value)
    mlflow_tracking_server_arn_value = resolve_mlflow_tracking_server_arn(
        mlflow_tracking_server_arn_value,
        mlflow_tracking_server_name_value,
    )
    assert_no_stale_local_code()
    assert_no_remote_function_steps()
    print(f"Using pipeline.py from: {Path(__file__).resolve()}")
    if mlflow_tracking_server_arn_value != "__disabled__":
        print(f"Logging MLflow runs to: {mlflow_tracking_server_arn_value}")
    else:
        print(
            "MLflow tracking server not configured; writing classic/S3 experiment records only."
        )
    print(f"Upserting pipeline: {pipeline.name}")
    pipeline.upsert(role_arn=role)

    execution = pipeline.start(
        parameters={
            "InputDataS3Uri": input_data_s3_uri,
            "TargetColumn": target_column_name,
            "ModelApprovalStatus": model_approval_status_value,
            "ExperimentName": experiment_name_value,
            "MlflowTrackingServerArn": mlflow_tracking_server_arn_value,
        }
    )
    print(f"Started execution: {execution.arn}")

    if wait:
        execution.wait()
        print(execution.describe())

    return execution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Upsert the pipeline and start a new execution.",
    )
    parser.add_argument(
        "--input-data",
        help="S3 URI for the raw input CSV, for example s3://bucket/path/data.csv.",
    )
    parser.add_argument(
        "--target-column",
        default="__last__",
        help="Target column name. Defaults to the last CSV column.",
    )
    parser.add_argument(
        "--model-approval-status",
        default="PendingManualApproval",
        help="Initial Model Registry approval status.",
    )
    parser.add_argument(
        "--experiment-name",
        default="xgboost-eks-experiments",
        help="Experiment name used for pipeline executions and MLflow runs.",
    )
    parser.add_argument(
        "--mlflow-tracking-server-arn",
        default="",
        help="Optional SageMaker managed MLflow tracking server ARN.",
    )
    parser.add_argument(
        "--mlflow-tracking-server-name",
        default="",
        help=(
            "Optional SageMaker managed MLflow tracking server name. "
            "The script resolves it to an ARN before starting the pipeline."
        ),
    )
    parser.add_argument(
        "--max-tuning-jobs",
        type=int,
        default=DEFAULT_MAX_TUNING_JOBS,
        help="Maximum number of candidate training jobs for SageMaker Automatic Model Tuning.",
    )
    parser.add_argument(
        "--max-parallel-tuning-jobs",
        type=int,
        default=DEFAULT_MAX_PARALLEL_TUNING_JOBS,
        help="Maximum number of tuning jobs SageMaker can run in parallel.",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for the pipeline execution to finish."
    )
    args = parser.parse_args()

    if args.submit:
        if not args.input_data:
            parser.error("--submit requires --input-data s3://bucket/path/file.csv")
        submit_pipeline(
            args.input_data,
            target_column_name=args.target_column,
            model_approval_status_value=args.model_approval_status,
            experiment_name_value=args.experiment_name,
            mlflow_tracking_server_arn_value=args.mlflow_tracking_server_arn,
            mlflow_tracking_server_name_value=args.mlflow_tracking_server_name,
            max_tuning_jobs_value=args.max_tuning_jobs,
            max_parallel_tuning_jobs_value=args.max_parallel_tuning_jobs,
            wait=args.wait,
        )
    else:
        assert_no_stale_local_code()
        assert_no_remote_function_steps()
        print(f"Using pipeline.py from: {Path(__file__).resolve()}")
        print(
            "Pipeline object built. Add --submit --input-data s3://bucket/path/file.csv to upsert and start it."
        )
