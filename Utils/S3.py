# S3 helpers. Error messages are mapped to plain language so the UI never
# dumps raw botocore traces (which can embed bucket policies or ARNs), and
# credentials never appear anywhere in this module.

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pathlib import Path

import pandas as pd

from Utils.logsys import get_logger
from Utils.paths import AIConversionRequired, read_tabular, SUPPORTED_DATASET_EXTENSIONS


logger = get_logger("S3")

# defensive stop so a runaway bucket listing cannot hang the page forever
MAX_S3_OBJECTS_SCAN = 20000


def get_s3_client(aws_access_key, aws_secret_key, region_name="us-east-1"):
    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )


def describe_s3_error(error):
    """Translate a boto3 failure into a user-safe message (no secrets/ARNs)."""
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        messages = {
            "AccessDenied": "AWS denied access. Check the key/secret and the "
                            "bucket policy allows s3:ListBucket / s3:GetObject.",
            "InvalidAccessKeyId": "That AWS Access Key ID was not recognized.",
            "SignatureDoesNotMatch": "The AWS secret key does not match the access key.",
            "NoSuchBucket": "No such bucket exists in this region/account.",
            "NoSuchKey": "That object no longer exists in the bucket.",
            "AuthorizationHeaderMalformed": "Credentials were malformed; re-enter them.",
            "BucketAlreadyExists": "That bucket name is taken.",
        }
        if code in messages:
            return messages[code]
        logger.warning("s3 ClientError code=%s status=%s", code, status)
        return f"AWS rejected the request ({code or 'unknown error'})."
    if isinstance(error, BotoCoreError):
        logger.warning("s3 BotoCoreError: %s: %s", type(error).__name__, error)
        return "Could not reach Amazon S3. Check the region name and your network."
    logger.warning("s3 unexpected error: %s: %s", type(error).__name__, error)
    return "An unexpected AWS error occurred."


def list_s3_datasets(bucket_name, client):
    suffixes = tuple(sorted(SUPPORTED_DATASET_EXTENSIONS))
    files = []
    request = {"Bucket": bucket_name}
    scanned = 0

    while True:
        response = client.list_objects_v2(**request)
        for obj in response.get("Contents", []):
            scanned += 1
            key = obj["Key"]
            if key.lower().endswith(suffixes):
                files.append(key)

        if not response.get("IsTruncated") or scanned >= MAX_S3_OBJECTS_SCAN:
            break

        next_token = response.get("NextContinuationToken")
        if not next_token:
            break
        request["ContinuationToken"] = next_token

    return files


def download_s3_dataset(bucket_name, file_key, client):
    obj = client.get_object(Bucket=bucket_name, Key=file_key)
    body = obj["Body"].read()
    try:
        df = read_tabular(body, filename=Path(file_key).name)
    except AIConversionRequired:
        # unparseable formats still return the raw bytes so callers can save the file
        return None, body
    return df, body


def upload_s3_dataset(df, bucket_name, file_key, client):
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    client.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=csv_bytes,
        ContentType="text/csv"
    )
    return True
