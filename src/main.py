"""Main entry point for the song processing worker."""

import logging
import sys
from pathlib import Path

# Add project root to path for imports when running directly
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import Config, load_config
from src.processor import BaseProcessor
from src.processors.dummy_processor import DummyProcessor
from src.processors.musicxml_processor import MusicXMLProcessor
from src.s3_client import make_s3_client
from src.worker import create_worker


def setup_logging() -> None:
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_processor(config: Config) -> BaseProcessor:
    """
    Create the processor instance based on configuration.

    In a more advanced setup, this could be configured via
    environment variables to select different processors.

    Args:
        config: Application configuration.

    Returns:
        Processor instance for job handling.
    """
    s3_client = make_s3_client(
        endpoint_url=config.s3_endpoint_url,
        access_key_id=config.s3_access_key_id,
        secret_access_key=config.s3_secret_access_key,
        region_name=config.s3_region,
        force_path_style=config.s3_force_path_style,
    )

    # Use MusicXML processor as default
    # For testing, you could switch to DummyProcessor
    return MusicXMLProcessor(s3_client=s3_client, bucket=config.s3_bucket)


def main() -> None:
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting song processing worker...")

    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration for queue: {config.queue}")

    # Create processor
    processor = create_processor(config)
    logger.info(f"Using processor: {processor.get_name()}")

    # Create and start worker
    worker = create_worker(config=config, processor=processor)
    worker.start()


if __name__ == "__main__":
    main()
