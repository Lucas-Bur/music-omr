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
    conn = psycopg.connect(db_url, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def claim_job(conn: psycopg.Connection, job_id: str) -> Optional[Dict[str, Any]]:
    """
    Claim a pending job for processing.

    Idempotency: Only claims jobs that are still PENDING.

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
          started_at = COALESCE(started_at, NOW()),
          error_message = NULL
      WHERE id = %(id)s
        AND status = 'PENDING'
      RETURNING id, status, started_at;
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
          finished_at = NOW(),
          error_message = %(err)s
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
          finished_at = NOW(),
          error_message = NULL
      WHERE id = %(id)s;
      """,
            {"id": job_id},
        )


def fetch_inputs(conn: psycopg.Connection, job_id: str) -> Dict[str, Any]:
    """
    Fetch input file information for a job.

    Args:
        conn: Database connection.
        job_id: The job identifier.

    Returns:
        Dict with input file information.

    Raises:
        RuntimeError: If no input file found.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
      SELECT f.s3_key
      FROM files f
      WHERE f.song_id = %(id)s
        AND f.file_type = 'ORIGINAL_PDF'
      ORDER BY f.created_at ASC
      LIMIT 1;
      """,
            {"id": job_id},
        )
        row = cur.fetchone()

    if not row:
        raise RuntimeError("No ORIGINAL_PDF input file for job")

    return {"pdf_key": row["s3_key"]}


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
      INSERT INTO files (song_id, file_type, s3_key, size_bytes)
      VALUES (%(song_id)s, %(file_type)s, %(s3_key)s, %(size_bytes)s);
      """,
            {
                "song_id": job_id,
                "file_type": file_type,
                "s3_key": s3_key,
                "size_bytes": size,
            },
        )
