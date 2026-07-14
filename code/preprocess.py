import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--target-column", default="__last__")
    args = parser.parse_args()

    parsed = urlparse(args.input_data)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    df = pd.read_csv(obj["Body"])

    if df.empty:
        raise ValueError("Input CSV is empty.")

    target_column = (
        df.columns[-1] if args.target_column == "__last__" else args.target_column
    )
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
        )

    y = pd.to_numeric(df[target_column], errors="raise")
    x = df.drop(columns=[target_column])
    x = pd.get_dummies(x, dummy_na=True, dtype=int)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)
    x.insert(0, target_column, y)

    train_df, temp_df = train_test_split(x, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    outputs = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    for channel_name, split_df in outputs.items():
        output_dir = Path("/opt/ml/processing") / channel_name
        output_dir.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(output_dir / "data.csv", index=False, header=False)
        print(
            f"Wrote {channel_name}: {split_df.shape[0]} rows, {split_df.shape[1]} columns"
        )

    analysis_dir = Path("/opt/ml/processing/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(analysis_dir / "data.csv", index=False, header=True)

    feature_columns = [column for column in x.columns if column != target_column]
    baseline = train_df[feature_columns].median(numeric_only=True).fillna(0)
    baseline.to_frame().T.to_csv(
        analysis_dir / "baseline.csv", index=False, header=False
    )
    (analysis_dir / "headers.json").write_text(
        json.dumps(
            {
                "label": target_column,
                "headers": list(x.columns),
                "feature_headers": feature_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote analysis dataset: {test_df.shape[0]} rows, {test_df.shape[1]} columns"
    )


if __name__ == "__main__":
    main()
