"""Message queue worker for processing jobs."""

import json
import sys
import time
import traceback
from typing import Any, Callable, Dict, Optional

import psycopg

import pika

from src.config import Config
from src.db import (
    claim_job,
    db_connect,
    insert_output_file,
    mark_completed,
    mark_failed,
)
from src.processor import BaseProcessor, ProcessingResult
from src.s3_client import make_s3_client
from src.task_types import validate_task_message


class JobWorker:
    """
    Worker that processes jobs from a message queue.

    The worker handles message consumption, job claiming,
    processing, and status updates.
    """

    def __init__(
        self,
        config: Config,
        processor: BaseProcessor,
        amqp_url: Optional[str] = None,
    ):
        """
        Initialize the job worker.

        Args:
            config: Application configuration.
            processor: Processor instance to use for jobs.
            amqp_url: Override AMQP URL from config.
        """
        self.config = config
        self.processor = processor
        self.amqp_url = amqp_url or config.amqp_url

        self.s3_client = make_s3_client(
            endpoint_url=config.s3_endpoint_url,
            access_key_id=config.s3_access_key_id,
            secret_access_key=config.s3_secret_access_key,
            region_name=config.s3_region,
            force_path_style=config.s3_force_path_style,
        )

    def start(self) -> None:
        """Start consuming messages from the queue."""
        params = pika.URLParameters(self.amqp_url)
        params.heartbeat = 600  # 10 minutes to handle long-running tasks

        while True:
            try:
                connection = pika.BlockingConnection(params)
                channel = connection.channel()

                # Declare queue (ensure it exists)
                channel.queue_declare(queue=self.config.queue, durable=True)

                # Process one message at a time for serial processing
                channel.basic_qos(prefetch_count=1)

                # Set up consumer
                channel.basic_consume(
                    queue=self.config.queue,
                    on_message_callback=self._create_message_handler(),
                    auto_ack=False,
                )

                print(f"[worker] consuming queue={self.config.queue}", flush=True)
                channel.start_consuming()

            except KeyboardInterrupt:
                print("[worker] stopping", flush=True)
                sys.exit(0)
            except Exception as e:
                print(f"[worker] connection error: {e}", flush=True)
                time.sleep(2.0)

    def _create_message_handler(self) -> Callable:
        """Create the message handler bound to this worker instance."""

        def on_message(
            channel: Any,
            method: Any,
            properties: Any,
            body: bytes,
        ) -> None:
            self._handle_message(channel, method, body)

        return on_message

    def _handle_message(
        self,
        channel: Any,
        method: Any,
        body: bytes,
    ) -> None:
        """
        Handle an incoming message from the queue.

        Args:
            channel: RabbitMQ channel.
            method: Delivery method.
            body: Message body bytes.
        """
        try:
            message_data = self._parse_message(body)
            print(message_data)
            job_id = message_data["job_id"]
            task_type = message_data["task_type"]
            task_params = message_data["task_params"]
            print(
                f"[debug] parsed message: job_id={job_id}, task_type={task_type}, task_params={task_params}",
                flush=True,
            )
        except Exception as e:
            # Malformed message - don't requeue to avoid infinite loop
            print(e)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        with db_connect(self.config.db_url) as conn:
            try:
                print(f"[debug] claiming job")
                # Transaction 1: Claim the job
                conn.execute("BEGIN;")
                claimed = self._claim_job(conn, job_id)
                conn.commit()

                if not claimed:
                    # Job already processing or completed - ack and skip
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                    return

                # Process outside transaction
                progress_callback = self._create_progress_callback(conn, job_id)
                result = self._process_task(
                    job_id, task_type, task_params, progress_callback
                )

                # Transaction: Complete only database updates
                conn.execute("BEGIN;")
                self._complete_task(conn, job_id, result)
                conn.commit()

                channel.basic_ack(delivery_tag=method.delivery_tag)

            except Exception as e:
                print(e)
                self._handle_error(conn, channel, method, job_id, e)

    def _parse_message(self, body: bytes) -> Dict[str, Any]:
        """Parse task message from message body."""
        data = json.loads(body.decode("utf-8"))

        # Validate message structure
        validate_task_message(data)

        return data

    def _claim_job(
        self, conn: psycopg.Connection, job_id: str
    ) -> Optional[Dict[str, Any]]:
        """Claim a pending job for processing."""
        return claim_job(conn, job_id)

    def _create_progress_callback(self, conn: psycopg.Connection, job_id: str):
        """Create a progress callback that updates the database."""

        def update_progress(percentage: int):
            with conn.cursor() as cur:
                conn.execute("BEGIN;")
                cur.execute(
                    "UPDATE songs SET progress = %s WHERE id = %s", (percentage, job_id)
                )
                conn.commit()

        return update_progress

    def _process_task(
        self,
        job_id: str,
        task_type: str,
        task_params: Dict[str, Any],
        progress_callback,
    ) -> ProcessingResult:
        """
        Process a task without database operations.

        Args:
            job_id: Job identifier.
            task_type: Type of the task to process.
            task_params: Task-specific parameters.
            progress_callback: Callback to report progress.

        Returns:
            ProcessingResult from the processor.
        """
        # Fetch task-specific input files (no DB needed here)
        inputs = self._fetch_task_inputs(None, job_id, task_type, task_params)
        print(f"[debug] fetched inputs: {inputs}", flush=True)

        # Process using the configured processor
        result = self.processor.process(job_id, inputs, progress_callback)
        print(
            f"[debug] processing result: success={result.success}, output_files={len(result.output_files)}",
            flush=True,
        )

        if not result.success:
            raise RuntimeError(result.error_message or "Processing failed")

        return result

    def _complete_task(
        self,
        conn: psycopg.Connection,
        job_id: str,
        result: ProcessingResult,
    ) -> None:
        """
        Complete a task by inserting files and marking as completed.

        Args:
            conn: Database connection.
            job_id: Job identifier.
            result: Processing result with output files.
        """
        # Insert output file records
        for output in result.output_files:
            insert_output_file(
                conn,
                job_id,
                file_type=output["file_type"],
                s3_key=output["s3_key"],
                size=output["size_bytes"],
            )

        # Mark job as completed
        mark_completed(conn, job_id)

    def _fetch_task_inputs(
        self,
        conn: Optional[psycopg.Connection],
        job_id: str,
        task_type: str,
        task_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract input file information from task parameters.

        Args:
            conn: Database connection (unused, kept for interface compatibility).
            job_id: Job identifier (unused, kept for interface compatibility).
            task_type: Type of the task (unused, kept for interface compatibility).
            task_params: Task parameters containing input_key.

        Returns:
            Dict with input file information.

        Raises:
            RuntimeError: If input_key not found in task_params.
        """
        input_key = task_params.get("input_key")
        if not input_key:
            raise RuntimeError("task_params missing required field: input_key")

        return {"input_key": input_key}

    def _handle_error(
        self,
        conn: Optional[psycopg.Connection],
        channel: Any,
        method: Any,
        job_id: str,
        error: Exception,
    ) -> None:
        """
        Handle processing errors.

        Args:
            conn: Database connection (may be None).
            channel: RabbitMQ channel.
            method: Delivery method.
            job_id: Job identifier.
            error: The exception that occurred.
        """
        print(f"[debug] handling error for job {job_id}: {error}", flush=True)
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass

            try:
                conn.execute("BEGIN;")
                mark_failed(conn, job_id, error_message=traceback.format_exc())
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass

        # Don't requeue to avoid poison message loops
        try:
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            print(f"[error] failed to nack message: {e}", flush=True)


def create_worker(config: Config, processor: BaseProcessor) -> JobWorker:
    """
    Factory function to create a worker instance.

    Args:
        config: Application configuration.
        processor: Processor for job handling.

    Returns:
        Configured JobWorker instance.
    """
    return JobWorker(config=config, processor=processor)
