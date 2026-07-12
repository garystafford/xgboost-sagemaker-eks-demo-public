import argparse
import json
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-uri", required=True)
    parser.add_argument("--test-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    s3 = boto3.client("s3")

    parsed_model = urlparse(args.model_s3_uri)
    model_tar_path = "/tmp/model.tar.gz"
    model_dir = Path("/tmp/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    s3.download_file(parsed_model.netloc, parsed_model.path.lstrip("/"), model_tar_path)
    with tarfile.open(model_tar_path) as tar:
        tar.extractall(model_dir)

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "xgboost-model"))

    test_df = pd.read_csv(Path(args.test_data_dir) / "data.csv", header=None)
    y_test = test_df.iloc[:, 0].values
    x_test = test_df.iloc[:, 1:].values
    predictions = booster.predict(xgb.DMatrix(x_test))

    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - np.sum((y_test - predictions) ** 2) / denominator)

    report = {
        "regression_metrics": {
            "rmse": {"value": rmse},
            "r2_score": {"value": r2},
        },
        "model_s3_uri": args.model_s3_uri,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
