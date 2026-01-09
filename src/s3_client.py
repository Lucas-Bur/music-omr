"""S3 client operations for file storage."""

from typing import Any

import boto3
from botocore.config import Config as BotoConfig


def make_s3_client(
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    region_name: str,
    force_path_style: bool = True,
) -> Any:
    """
    Create an S3 client with the specified configuration.

    Args:
        endpoint_url: S3 endpoint URL (e.g., http://localhost:9000).
        access_key_id: AWS access key ID.
        secret_access_key: AWS secret access key.
        region_name: AWS region name.
        force_path_style: Whether to use path-style addressing.

    Returns:
        A boto3 S3 client instance.
    """
    s3_config = BotoConfig(s3={"addressing_style": "path"} if force_path_style else {})

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region_name,
        config=s3_config,
    )


def download_s3_object(s3_client: Any, bucket: str, key: str) -> bytes:
    """
    Download an S3 object as bytes.

    Args:
        s3_client: boto3 S3 client.
        bucket: S3 bucket name.
        key: Object key.

    Returns:
        The object data as bytes.
    """
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def upload_bytes_to_s3(
    s3_client: Any,
    bucket: str,
    key: str,
    data: bytes,
    content_type: str,
) -> None:
    """
    Upload bytes to S3.

    Args:
        s3_client: boto3 S3 client.
        bucket: S3 bucket name.
        key: Object key.
        data: Bytes to upload.
        content_type: MIME type of the content.
    """
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def get_s3_object_size(s3_client: Any, bucket: str, key: str) -> int:
    """
    Get the size of an S3 object in bytes.

    Args:
        s3_client: boto3 S3 client.
        bucket: S3 bucket name.
        key: Object key.

    Returns:
        Object size in bytes.
    """
    resp = s3_client.head_object(Bucket=bucket, Key=key)
    return resp["ContentLength"]
