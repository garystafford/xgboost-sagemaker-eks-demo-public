import argparse

import boto3
from botocore.exceptions import ClientError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-uri", required=True)
    parser.add_argument("--metrics-s3-uri", required=True)
    parser.add_argument("--approval-status", required=True)
    parser.add_argument("--inference-image-uri", required=True)
    parser.add_argument("--model-package-group-name", required=True)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()

    sm_client = boto3.client("sagemaker", region_name=args.region)

    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=args.model_package_group_name,
            ModelPackageGroupDescription="XGBoost regression models for EKS deployment",
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        error_message = exc.response.get("Error", {}).get("Message", "")
        if error_code != "ValidationException" or "already exists" not in error_message:
            raise

    response = sm_client.create_model_package(
        ModelPackageGroupName=args.model_package_group_name,
        ModelApprovalStatus=args.approval_status,
        InferenceSpecification={
            "Containers": [
                {
                    "Image": args.inference_image_uri,
                    "ModelDataUrl": args.model_s3_uri,
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": args.metrics_s3_uri,
                }
            }
        },
    )

    print(response["ModelPackageArn"])


if __name__ == "__main__":
    main()
