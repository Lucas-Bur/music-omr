"""Message queue worker for processing jobs."""

import json
import sys
import time
import traceback
from typing import Any, Callable, Dict, Optional

import pika

from src.config import Config
from src.db import (
    db_connect,
    fetch_inputs,
    insert_output_file,
    mark_completed,
    mark_failed,
)
from src.processor import BaseProcessor
from src.s3_client import make_s3_client


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
            job_id = self._parse_message(body)
        except Exception:
            # Malformed message - don't requeue to avoid infinite loop
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        conn = None
        try:
            conn = db_connect(self.config.db_url)

            # Transaction 1: Claim the job
            conn.execute("BEGIN;")
            claimed = self._claim_job(conn, job_id)
            conn.commit()

            if not claimed:
                # Job already processing or completed - ack and skip
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return

            # Transaction 2: Process and complete
            conn.execute("BEGIN;")
            self._process_and_complete(conn, job_id)
            conn.commit()

            channel.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            self._handle_error(conn, channel, method, job_id, e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _parse_message(self, body: bytes) -> str:
        """Parse job_id from message body."""
        data = json.loads(body.decode("utf-8"))
        job_id = data.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError("Message missing job_id")
        return job_id

    def _claim_job(self, conn: Any, job_id: str) -> Optional[Dict[str, Any]]:
        """Claim a pending job for processing."""
        from src.db import claim_job

        return claim_job(conn, job_id)

    def _process_and_complete(self, conn: Any, job_id: str) -> None:
        """
        Process a job and mark it as completed.

        Args:
            conn: Database connection.
            job_id: Job identifier.
        """
        # Fetch input files
        inputs = fetch_inputs(conn, job_id)

        # Process using the configured processor
        result = self.processor.process(job_id, inputs)

        if not result.success:
            raise RuntimeError(result.error_message or "Processing failed")

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

    def _handle_error(
        self,
        conn: Any,
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
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


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
