import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse


TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}
ClientError = None


def load_boto3():
    import boto3
    from botocore.exceptions import ClientError as boto_client_error

    return boto3, boto_client_error


def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected S3 URI, got: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def upload_json(s3_client, s3_uri, payload):
    bucket, key = parse_s3_uri(s3_uri)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_baseline(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Clarify baseline.csv is empty.")
    return [[float(value) for value in text.split(",")]]


def create_model(sm_client, model_name, role_arn, image_uri, model_s3_uri):
    try:
        sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": model_s3_uri,
            },
        )
    except ClientError as exc:
        error = exc.response.get("Error", {})
        message = error.get("Message", "")
        if (
            error.get("Code") != "ValidationException"
            or "already exists" not in message
        ):
            raise


def delete_model(sm_client, model_name):
    try:
        sm_client.delete_model(ModelName=model_name)
    except ClientError as exc:
        error = exc.response.get("Error", {})
        if error.get("Code") != "ValidationException":
            raise


def start_clarify_job(
    sm_client,
    job_name,
    role_arn,
    clarify_image_uri,
    analysis_config_s3_uri,
    analysis_data_s3_uri,
    output_s3_uri,
    instance_type,
):
    request = {
        "ProcessingJobName": job_name,
        "RoleArn": role_arn,
        "AppSpecification": {"ImageUri": clarify_image_uri},
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 30,
            }
        },
        "ProcessingInputs": [
            {
                "InputName": "analysis_config",
                "S3Input": {
                    "S3Uri": analysis_config_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/config",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            {
                "InputName": "dataset",
                "S3Input": {
                    "S3Uri": analysis_data_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "analysis_result",
                    "S3Output": {
                        "S3Uri": output_s3_uri,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    }

    try:
        sm_client.create_processing_job(**request)
    except ClientError as exc:
        error = exc.response.get("Error", {})
        message = error.get("Message", "")
        if (
            error.get("Code") != "ValidationException"
            or "already exists" not in message
        ):
            raise
        print(f"Clarify processing job {job_name} already exists; waiting for it.")


def wait_for_processing_job(sm_client, job_name):
    while True:
        response = sm_client.describe_processing_job(ProcessingJobName=job_name)
        status = response["ProcessingJobStatus"]
        print(f"Clarify processing job {job_name} status: {status}")
        if status in TERMINAL_STATUSES:
            if status != "Completed":
                reason = response.get("FailureReason", "No failure reason provided.")
                raise RuntimeError(
                    f"Clarify processing job {job_name} ended with {status}: {reason}"
                )
            return response
        time.sleep(30)


def main():
    global ClientError

    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-data-dir", required=True)
    parser.add_argument("--analysis-data-s3-uri", required=True)
    parser.add_argument("--model-s3-uri", required=True)
    parser.add_argument("--inference-image-uri", required=True)
    parser.add_argument("--clarify-image-uri", required=True)
    parser.add_argument("--clarify-output-s3-uri", required=True)
    parser.add_argument("--summary-output-dir", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--clarify-job-name", required=True)
    parser.add_argument("--clarify-instance-type", default="ml.m5.xlarge")
    args = parser.parse_args()
    boto3, ClientError = load_boto3()

    analysis_dir = Path(args.analysis_data_dir)
    headers_payload = read_json(analysis_dir / "headers.json")
    baseline = read_baseline(analysis_dir / "baseline.csv")

    analysis_config = {
        "dataset_type": "text/csv",
        "headers": headers_payload["headers"],
        "label": headers_payload["label"],
        "methods": {
            "shap": {
                "baseline": baseline,
                "num_samples": 100,
                "agg_method": "mean_abs",
                "save_local_shap_values": True,
            }
        },
        "predictor": {
            "model_name": args.model_name,
            "instance_type": args.clarify_instance_type,
            "initial_instance_count": 1,
            "accept_type": "text/csv",
            "content_type": "text/csv",
        },
    }

    s3_client = boto3.client("s3", region_name=args.region)
    sm_client = boto3.client("sagemaker", region_name=args.region)

    config_s3_uri = (
        f"{args.clarify_output_s3_uri.rstrip('/')}/config/analysis_config.json"
    )
    analysis_output_s3_uri = f"{args.clarify_output_s3_uri.rstrip('/')}/analysis"
    analysis_report_s3_uri = f"{analysis_output_s3_uri}/analysis.json"
    upload_json(s3_client, config_s3_uri, analysis_config)

    create_model(
        sm_client,
        args.model_name,
        args.role_arn,
        args.inference_image_uri,
        args.model_s3_uri,
    )

    try:
        start_clarify_job(
            sm_client,
            args.clarify_job_name,
            args.role_arn,
            args.clarify_image_uri,
            config_s3_uri,
            args.analysis_data_s3_uri,
            analysis_output_s3_uri,
            args.clarify_instance_type,
        )
        response = wait_for_processing_job(sm_client, args.clarify_job_name)
    finally:
        delete_model(sm_client, args.model_name)

    summary = {
        "clarify_job_name": args.clarify_job_name,
        "clarify_model_name": args.model_name,
        "clarify_output_s3_uri": args.clarify_output_s3_uri,
        "analysis_config_s3_uri": config_s3_uri,
        "analysis_report_s3_uri": analysis_report_s3_uri,
        "processing_job_arn": response["ProcessingJobArn"],
        "methods": ["shap"],
        "feature_count": len(headers_payload["feature_headers"]),
    }

    output_dir = Path(args.summary_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clarify_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
