"""
Abstract base class for job processors.

This module defines the processor interface that all concrete processors
must implement. This enables pluggable processing logic and easy testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProcessingResult:
    """Result of a processing job."""

    success: bool
    output_files: List[Dict[str, Any]]
    error_message: Optional[str] = None


class BaseProcessor(ABC):
    """
    Abstract base class for job processors.

    All processors must implement the process method. This follows
    the Strategy pattern, allowing different processing algorithms
    to be used interchangeably.
    """

    @abstractmethod
    def process(self, job_id: str, inputs: Dict[str, Any]) -> ProcessingResult:
        """
        Process a job with the given inputs.

        Args:
            job_id: Unique identifier for the job.
            inputs: Dictionary of input data (e.g., file keys).

        Returns:
            ProcessingResult with success status and output files.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the processor name/identifier.

        Returns:
            Processor name string.
        """
        pass

    @property
    @abstractmethod
    def supported_input_types(self) -> List[str]:
        """
        Get list of supported input file types.

        Returns:
            List of file type strings.
        """
        pass
