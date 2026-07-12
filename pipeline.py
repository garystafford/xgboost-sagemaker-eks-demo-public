import argparse
from pathlib import Path

from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)
from sagemaker.core.training.configs import Compute, InputData, OutputDataConfig
from sagemaker.core.workflow.functions import JsonGet, Join
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.train import ModelTrainer


CODE_DIR = Path(__file__).resolve().parent / "code"

pipeline_session = PipelineSession()
role = get_execution_role()
region = pipeline_session.boto_region_name
bucket = pipeline_session.default_bucket()

pipeline_prefix = "xgboost-eks-pipeline"
model_prefix = f"{pipeline_prefix}/models"

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

input_data_uri = ParameterString(name="InputDataS3Uri")
target_column = ParameterString(name="TargetColumn", default_value="__last__")
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
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
    ],
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    step_args=preprocess_args,
)


# --- Step 2: Train XGBoost with ModelTrainer + TrainingStep ---
model_trainer = ModelTrainer(
    training_image=xgboost_image,
    role=role,
    sagemaker_session=pipeline_session,
    base_job_name="xgboost-eks-train",
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

training_args = model_trainer.train(
    input_data_config=[
        InputData(
            channel_name="train",
            data_source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
        InputData(
            channel_name="validation",
            data_source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    ],
    wait=False,
)

training_step = TrainingStep(
    name="TrainXGBoost",
    step_args=training_args,
    depends_on=[preprocess_step],
)

best_model_s3_uri = training_step.properties.ModelArtifacts.S3ModelArtifacts


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
    depends_on=[training_step],
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


# --- Step 4: Register with a ProcessingStep ---
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
    depends_on=[evaluation_step],
)

pipeline = Pipeline(
    name="xgboost-eks-pipeline-v3-processing",
    parameters=[input_data_uri, target_column, model_approval_status],
    steps=[preprocess_step, training_step, evaluation_step, register_step],
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


def submit_pipeline(
    input_data_s3_uri,
    target_column_name="__last__",
    model_approval_status_value="PendingManualApproval",
    wait=False,
):
    assert_no_stale_local_code()
    assert_no_remote_function_steps()
    print(f"Using pipeline.py from: {Path(__file__).resolve()}")
    print(f"Upserting pipeline: {pipeline.name}")
    pipeline.upsert(role_arn=role)

    execution = pipeline.start(
        parameters={
            "InputDataS3Uri": input_data_s3_uri,
            "TargetColumn": target_column_name,
            "ModelApprovalStatus": model_approval_status_value,
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
            wait=args.wait,
        )
    else:
        assert_no_stale_local_code()
        assert_no_remote_function_steps()
        print(f"Using pipeline.py from: {Path(__file__).resolve()}")
        print(
            "Pipeline object built. Add --submit --input-data s3://bucket/path/file.csv to upsert and start it."
        )
