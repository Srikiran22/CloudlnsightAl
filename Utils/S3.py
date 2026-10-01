# S3 helpers. Error messages are mapped to plain language so the UI never
# dumps raw botocore traces (which can embed bucket policies or ARNs), and
# credentials never appear anywhere in this module.

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pathlib import Path

from Utils.logsys import get_logger
from Utils.paths import (
    AIConversionRequired, MAX_UPLOAD_BYTES, read_tabular,
    SUPPORTED_DATASET_EXTENSIONS,
)


logger = get_logger("S3")

# defensive stop so a runaway bucket listing cannot hang the page forever
MAX_S3_OBJECTS_SCAN = 20000

# Bounded client network timeouts to prevent hanging spinners on unresponsive endpoints
S3_CLIENT_CONFIG = Config(
    connect_timeout=10,
    read_timeout=30,
    retries={"max_attempts": 2},
)


def get_s3_client(aws_access_key, aws_secret_key, region_name="us-east-1", aws_session_token=None):
    kwargs = {
        "aws_access_key_id": aws_access_key,
        "aws_secret_access_key": aws_secret_key,
        "region_name": region_name,
        "config": S3_CLIENT_CONFIG,
    }
    if aws_session_token:
        kwargs["aws_session_token"] = aws_session_token
    return boto3.client("s3", **kwargs)


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


def list_s3_datasets(bucket_name, client, return_meta=False):
    suffixes = tuple(sorted(SUPPORTED_DATASET_EXTENSIONS))
    files = []
    request = {"Bucket": bucket_name}
    scanned = 0
    truncated = False

    while True:
        response = client.list_objects_v2(**request)
        for obj in response.get("Contents", []):
            scanned += 1
            key = obj["Key"]
            if key.lower().endswith(suffixes):
                files.append(key)

        is_truncated = bool(response.get("IsTruncated"))
        if not is_truncated or scanned >= MAX_S3_OBJECTS_SCAN:
            if is_truncated and scanned >= MAX_S3_OBJECTS_SCAN:
                truncated = True
            break

        next_token = response.get("NextContinuationToken")
        if not next_token:
            break
        request["ContinuationToken"] = next_token

    if return_meta:
        return files, {"truncated": truncated, "scanned": scanned, "max_scanned": MAX_S3_OBJECTS_SCAN}
    return files


def download_s3_dataset(bucket_name, file_key, client):
    obj = client.get_object(Bucket=bucket_name, Key=file_key)
    stream = obj.get("Body")
    try:
        content_length = obj.get("ContentLength")
        # same resource guard as local uploads: never stream an unbounded object
        # into memory just because the bucket holds it
        if content_length and content_length > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"S3 object is about {content_length / (1024 * 1024):.0f} MB; the "
                f"ingest limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Download "
                "it manually and trim the file first."
            )

        # Stream with a hard byte limit -- remains bounded even if ContentLength is missing
        chunks = []
        bytes_read = 0
        chunk_size = 64 * 1024

        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"S3 download exceeded the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit while streaming."
                )
            chunks.append(chunk)

        body = b"".join(chunks)
        try:
            df = read_tabular(body, filename=Path(file_key).name)
        except AIConversionRequired:
            # unparseable formats still return the raw bytes so callers can save the file
            return None, body
        return df, body
    finally:
        if stream is not None and hasattr(stream, "close"):
            try:
                stream.close()
            except Exception:
                pass


def upload_s3_dataset(df, bucket_name, file_key, client):
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    client.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=csv_bytes,
        ContentType="text/csv"
    )
    return True
