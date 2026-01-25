"""Database operations for the song processing worker."""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import psycopg
from psycopg.rows import dict_row


@contextmanager
def db_connect(db_url: str) -> Generator[psycopg.Connection, None, None]:
    """
    Context manager for database connections.

    Args:
        db_url: PostgreSQL connection URL.

    Yields:
        A database connection with dict row factory.
    """
    print(f"[debug] trying to make a connection @ {db_url}")
    conn = psycopg.connect(db_url, row_factory=dict_row)
    try:
        print(f"[debug] trying to yield connection")
        yield conn
    except Exception as e:
        print(f"[error] {e}")
    finally:
        print(f"[debug] closing connection")

        conn.close()


def claim_job(conn: psycopg.Connection, job_id: str) -> Optional[Dict[str, Any]]:
    """
    Claim a job for processing.

    Idempotency: Only claims jobs that are not currently PROCESSING.

    Args:
        conn: Database connection.
        job_id: The job identifier.

    Returns:
        The job row if claimed, None if not claimable.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
      UPDATE songs
      SET status = 'PROCESSING',
          "startedAt" = COALESCE("startedAt", NOW()),
          "errorMessage" = NULL,
          progress = 0
      WHERE id = %(id)s
        AND status != 'PROCESSING'
      RETURNING id, status, "startedAt", progress;
      """,
            {"id": job_id},
        )
        row = cur.fetchone()
        return row


def mark_failed(conn: psycopg.Connection, job_id: str, error_message: str) -> None:
    """
    Mark a job as failed with error details.

    Args:
        conn: Database connection.
        job_id: The job identifier.
        error_message: Error description (truncated to 10000 chars).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
      UPDATE songs
      SET status = 'FAILED',
          "finishedAt" = NOW(),
          "errorMessage" = %(err)s
      WHERE id = %(id)s;
      """,
            {"id": job_id, "err": error_message[:10000]},
        )


def mark_completed(conn: psycopg.Connection, job_id: str) -> None:
    """
    Mark a job as completed.

    Args:
        conn: Database connection.
        job_id: The job identifier.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
      UPDATE songs
      SET status = 'COMPLETED',
          "finishedAt" = NOW(),
          "errorMessage" = NULL,
          progress = 100
      WHERE id = %(id)s;
      """,
            {"id": job_id},
        )


def insert_output_file(
    conn: psycopg.Connection, job_id: str, file_type: str, s3_key: str, size: int
) -> None:
    """
    Insert an output file record into the database.

    Args:
        conn: Database connection.
        job_id: The song/job identifier.
        file_type: Type of the output file.
        s3_key: S3 object key.
        size: File size in bytes.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
      INSERT INTO files ("songId", "fileType", "s3Key", "sizeBytes")
      VALUES (%(song_id)s, %(file_type)s, %(s3_key)s, %(size_bytes)s);
      """,
            {
                "song_id": job_id,
                "file_type": file_type,
                "s3_key": s3_key,
                "size_bytes": size,
            },
        )
