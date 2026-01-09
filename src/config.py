"""Configuration management for the song processing worker."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    amqp_url: str
    queue: str
    db_url: str
    s3_endpoint_url: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket: str
    s3_region: str
    s3_force_path_style: bool


def getenv(key: str, default: str) -> str:
    """Get environment variable with fallback."""
    return os.getenv(key, default)


def load_config() -> Config:
    """Load configuration from environment variables with sensible defaults."""
    amqp_url = getenv("AMQP_URL", "amqp://app:app-secret@localhost:5672")
    queue = getenv("AMQP_QUEUE", "song_jobs")
    db_url = getenv("DATABASE_URL", "")

    s3_endpoint_url = getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    s3_access_key_id = getenv("S3_ACCESS_KEY_ID", "minioadmin")
    s3_secret_access_key = getenv("S3_SECRET_ACCESS_KEY", "minioadmin")
    s3_bucket = getenv("S3_BUCKET", "scores-bucket")
    s3_region = getenv("S3_REGION", "eu-central-1")
    s3_force_path_style = getenv("S3_FORCE_PATH_STYLE", "true").lower() in (
        "1",
        "true",
        "yes",
    )

    return Config(
        amqp_url=amqp_url,
        queue=queue,
        db_url=db_url,
        s3_endpoint_url=s3_endpoint_url,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        s3_bucket=s3_bucket,
        s3_region=s3_region,
        s3_force_path_style=s3_force_path_style,
    )
