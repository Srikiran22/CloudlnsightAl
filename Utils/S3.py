import boto3
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from Utils.paths import read_tabular, SUPPORTED_DATASET_EXTENSIONS


def get_s3_client(
    aws_access_key: str,
    aws_secret_key: str,
    region_name: str = "us-east-1"
):
    """Initialize and return an authenticated boto3 S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )


def list_s3_datasets(bucket_name: str, client) -> List[str]:
    """List all supported data files stored in an S3 bucket."""
    suffixes = tuple(sorted(SUPPORTED_DATASET_EXTENSIONS))
    files = []
    request = {"Bucket": bucket_name}

    while True:
        response = client.list_objects_v2(**request)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(suffixes):
                files.append(key)

        if not response.get("IsTruncated"):
            break

        next_token = response.get("NextContinuationToken")
        if not next_token:
            break
        request["ContinuationToken"] = next_token

    return files


def download_s3_dataset(bucket_name: str, file_key: str, client) -> Tuple[pd.DataFrame, bytes]:
    """Download a dataset from S3 as original bytes plus a parsed DataFrame."""
    obj = client.get_object(Bucket=bucket_name, Key=file_key)
    body = obj["Body"].read()
    df = read_tabular(body, filename=Path(file_key).name)
    return df, body


def upload_s3_dataset(
    df: pd.DataFrame,
    bucket_name: str,
    file_key: str,
    client
) -> bool:
    """Upload a Pandas DataFrame directly to S3 as a CSV file."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    
    client.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=csv_bytes,
        ContentType="text/csv"
    )
    return True
