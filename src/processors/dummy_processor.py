from typing import Any, Dict, List
from src.processor import BaseProcessor, ProcessingResult


class DummyProcessor(BaseProcessor):
    """
    A simple dummy processor for testing.

    Simulates processing without actual file operations.
    """

    def __init__(self):
        self.processed_jobs = []

    def get_name(self) -> str:
        return "dummy_processor"

    @property
    def supported_input_types(self) -> List[str]:
        return ["ORIGINAL_PDF"]

    def process(self, job_id: str, inputs: Dict[str, Any]) -> ProcessingResult:
        self.processed_jobs.append(job_id)
        print(f"Dummy processing job {job_id} with inputs: {inputs}")
        print(f"Processed jobs so far: {self.processed_jobs}")
        return ProcessingResult(
            success=True,
            output_files=[
                {
                    "file_type": "DUMMY_OUTPUT",
                    "s3_key": f"jobs/{job_id}/dummy.txt",
                    "size_bytes": 0,
                }
            ],
            error_message=None,
        )
