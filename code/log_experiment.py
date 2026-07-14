import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ClientError = None


def load_boto3():
    import boto3
    from botocore.exceptions import ClientError as boto_client_error

    return boto3, boto_client_error


def already_exists(exc):
    error = exc.response.get("Error", {})
    code = error.get("Code", "")
    message = error.get("Message", "")
    return code in {"ResourceInUse", "ResourceInUseException"} or (
        code == "ValidationException" and "already exists" in message.lower()
    )


def safe_component_name(display_name, trial_name):
    raw_name = f"{display_name}-{trial_name}"
    normalized = re.sub(r"[^A-Za-z0-9-]+", "-", raw_name).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        normalized = "promoted-model"

    if len(normalized) <= 120:
        return normalized

    digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
    return f"{normalized[:107].rstrip('-')}-{digest}"


def ensure_experiment(sm_client, experiment_name):
    try:
        sm_client.create_experiment(
            ExperimentName=experiment_name,
            DisplayName=experiment_name,
            Description="Pipeline executions for the XGBoost SageMaker to EKS demo.",
        )
    except ClientError as exc:
        if not already_exists(exc):
            raise


def ensure_trial(sm_client, experiment_name, trial_name):
    try:
        sm_client.create_trial(
            TrialName=trial_name,
            ExperimentName=experiment_name,
            DisplayName=trial_name,
        )
    except ClientError as exc:
        if not already_exists(exc):
            raise


def upsert_trial_component(
    sm_client, trial_component_name, display_name, parameters, artifacts
):
    now = datetime.now(timezone.utc)
    request = {
        "TrialComponentName": trial_component_name,
        "DisplayName": display_name,
        "Status": {
            "PrimaryStatus": "Completed",
            "Message": "Best HPO model selected, evaluated, and prepared for registry promotion.",
        },
        "StartTime": now,
        "EndTime": now,
        "Parameters": parameters,
        "InputArtifacts": artifacts["inputs"],
        "OutputArtifacts": artifacts["outputs"],
    }

    try:
        sm_client.create_trial_component(**request)
    except ClientError as exc:
        if not already_exists(exc):
            raise
        sm_client.update_trial_component(**request)


def associate_trial_component(sm_client, trial_name, trial_component_name):
    try:
        sm_client.associate_trial_component(
            TrialName=trial_name,
            TrialComponentName=trial_component_name,
        )
    except ClientError as exc:
        error = exc.response.get("Error", {})
        message = error.get("Message", "").lower()
        if "already associated" not in message and "already exists" not in message:
            raise


def number_parameter(value):
    return {"NumberValue": float(value)}


def string_parameter(value):
    return {"StringValue": str(value)}


def artifact(value, media_type):
    return {"Value": value, "MediaType": media_type}


def write_summary(output_dir, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary_path


def import_mlflow():
    try:
        import mlflow
        import sagemaker_mlflow  # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "mlflow>=3,<4",
                "sagemaker-mlflow>=0.5,<1",
            ]
        )
        import mlflow
        import sagemaker_mlflow  # noqa: F401

    return mlflow


def mlflow_enabled(tracking_server_arn):
    return tracking_server_arn and tracking_server_arn != "__disabled__"


def ensure_mlflow_dependencies(args):
    if not mlflow_enabled(args.mlflow_tracking_server_arn):
        return None

    return import_mlflow()


def log_to_mlflow(args, summary, summary_path, metrics, mlflow):
    if not mlflow:
        return None

    mlflow.set_tracking_uri(args.mlflow_tracking_server_arn)
    mlflow.set_experiment(args.mlflow_experiment_name or args.experiment_name)

    with mlflow.start_run(run_name=args.trial_name) as run:
        mlflow_summary = {
            "tracking_server_arn": args.mlflow_tracking_server_arn,
            "experiment_name": args.mlflow_experiment_name or args.experiment_name,
            "run_id": run.info.run_id,
        }
        summary["mlflow"] = mlflow_summary
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "model_package_group_name": args.model_package_group_name,
                "tuning_job_name": args.tuning_job_name,
                "best_training_job_name": args.best_training_job_name,
            }
        )
        tags = {
            "sagemaker.pipeline_trial_name": args.trial_name,
            "sagemaker.model_s3_uri": args.model_s3_uri,
            "sagemaker.evaluation_s3_uri": args.evaluation_s3_uri,
        }
        if args.clarify_report_s3_uri:
            tags["sagemaker.clarify_report_s3_uri"] = args.clarify_report_s3_uri
        mlflow.set_tags(tags)
        artifact_errors = []
        for local_path, artifact_path in [
            (args.evaluation_report_path, "evaluation"),
            (str(summary_path), "summary"),
        ]:
            try:
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
            except Exception as exc:  # noqa: BLE001
                artifact_errors.append(
                    {
                        "local_path": local_path,
                        "artifact_path": artifact_path,
                        "error": str(exc),
                    }
                )
                print(
                    f"Warning: unable to log MLflow artifact {local_path}: {exc}",
                    file=sys.stderr,
                )

        if artifact_errors:
            mlflow_summary["artifact_logging_errors"] = artifact_errors
            summary["mlflow"] = mlflow_summary
            summary_path.write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

        return mlflow_summary


def main():
    global ClientError

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mlflow-experiment-name")
    parser.add_argument("--mlflow-tracking-server-arn")
    parser.add_argument("--trial-name", required=True)
    parser.add_argument("--trial-component-display-name", required=True)
    parser.add_argument("--model-s3-uri", required=True)
    parser.add_argument("--evaluation-s3-uri", required=True)
    parser.add_argument("--clarify-report-s3-uri")
    parser.add_argument("--evaluation-report-path", required=True)
    parser.add_argument("--summary-output-dir", required=True)
    parser.add_argument("--summary-s3-uri", required=True)
    parser.add_argument("--model-package-group-name", required=True)
    parser.add_argument("--tuning-job-name", required=True)
    parser.add_argument("--best-training-job-name", required=True)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()

    mlflow = ensure_mlflow_dependencies(args)
    boto3, ClientError = load_boto3()

    evaluation_report = json.loads(
        Path(args.evaluation_report_path).read_text(encoding="utf-8")
    )
    regression_metrics = evaluation_report["regression_metrics"]
    rmse = regression_metrics["rmse"]["value"]
    r2_score = regression_metrics["r2_score"]["value"]

    trial_component_name = safe_component_name(
        args.trial_component_display_name,
        args.trial_name,
    )

    parameters = {
        "rmse": number_parameter(rmse),
        "r2_score": number_parameter(r2_score),
        "model_package_group_name": string_parameter(args.model_package_group_name),
        "tuning_job_name": string_parameter(args.tuning_job_name),
        "best_training_job_name": string_parameter(args.best_training_job_name),
    }

    artifacts = {
        "inputs": {
            "best_model_artifact": artifact(args.model_s3_uri, "s3/uri"),
            "evaluation_report": artifact(args.evaluation_s3_uri, "application/json"),
        },
        "outputs": {
            "experiment_summary": artifact(args.summary_s3_uri, "application/json"),
        },
    }

    summary = {
        "experiment_name": args.experiment_name,
        "trial_name": args.trial_name,
        "trial_component_name": trial_component_name,
        "model_package_group_name": args.model_package_group_name,
        "tuning_job_name": args.tuning_job_name,
        "best_training_job_name": args.best_training_job_name,
        "model_s3_uri": args.model_s3_uri,
        "evaluation_s3_uri": args.evaluation_s3_uri,
        "metrics": {
            "rmse": rmse,
            "r2_score": r2_score,
        },
    }
    if args.clarify_report_s3_uri:
        artifacts["inputs"]["clarify_explainability_report"] = artifact(
            args.clarify_report_s3_uri,
            "application/json",
        )
        summary["clarify_report_s3_uri"] = args.clarify_report_s3_uri

    output_dir = Path(args.summary_output_dir)
    summary_path = write_summary(output_dir, summary)

    sm_client = boto3.client("sagemaker", region_name=args.region)
    ensure_experiment(sm_client, args.experiment_name)
    ensure_trial(sm_client, args.experiment_name, args.trial_name)
    upsert_trial_component(
        sm_client,
        trial_component_name,
        args.trial_component_display_name,
        parameters,
        artifacts,
    )
    associate_trial_component(sm_client, args.trial_name, trial_component_name)

    mlflow_summary = log_to_mlflow(
        args,
        summary,
        summary_path,
        {"rmse": rmse, "r2_score": r2_score},
        mlflow,
    )
    if mlflow_summary:
        summary["mlflow"] = mlflow_summary
        write_summary(output_dir, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
