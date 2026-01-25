"""Task type definitions and constants."""

from enum import Enum
from typing import Dict, Any


class TaskType(Enum):
    """Supported task types for message processing."""

    GENERATE_XML_FROM_INPUT = "generate_xml_from_input"
    GENERATE_VOICES_FROM_XML = "generate_voices_from_xml"


class FileType(Enum):
    """Supported file types for input and output."""

    SCORE = "SCORE"
    MUSIC_XML = "MUSIC_XML"
    AUDIO = "AUDIO"


# Task type to processor name mapping
TASK_TYPE_TO_PROCESSOR = {
    TaskType.GENERATE_XML_FROM_INPUT: "xml_from_input_processor",
    TaskType.GENERATE_VOICES_FROM_XML: "voices_from_xml_processor",
}


def validate_task_message(message: Dict[str, Any]) -> None:
    """
    Validate a task message structure.

    Args:
        message: The message to validate.

    Raises:
        ValueError: If message structure is invalid.
    """
    required_fields = ["job_id", "task_type", "task_params"]

    for field in required_fields:
        if field not in message:
            raise ValueError(f"Message missing required field: {field}")

    # Validate task_type
    try:
        TaskType(message["task_type"])
    except ValueError:
        valid_types = [t.value for t in TaskType]
        raise ValueError(
            f"Invalid task_type: {message['task_type']}. Valid types: {valid_types}"
        )

    # Validate task_params structure
    task_params = message["task_params"]
    if not isinstance(task_params, dict):
        raise ValueError("task_params must be a dictionary")

    if "input_key" not in task_params:
        raise ValueError("task_params missing required field: input_key")

    if not isinstance(task_params["input_key"], str):
        raise ValueError("input_key must be a string")


def get_task_processor_name(task_type: str) -> str:
    """
    Get processor name for a task type.

    Args:
        task_type: The task type string.

    Returns:
        Processor name string.

    Raises:
        ValueError: If task_type is invalid.
    """
    try:
        enum_type = TaskType(task_type)
        return TASK_TYPE_TO_PROCESSOR[enum_type]
    except KeyError:
        raise ValueError(f"Unknown task_type: {task_type}")


def get_input_file_types_for_task(task_type: TaskType) -> list[str]:
    """
    Get supported input file types for a task type.

    Args:
        task_type: The task type enum.

    Returns:
        List of supported file types as strings.
    """
    if task_type == TaskType.GENERATE_XML_FROM_INPUT:
        return [FileType.SCORE.value]
    elif task_type == TaskType.GENERATE_VOICES_FROM_XML:
        return [FileType.MUSIC_XML.value]
    else:
        return []


def get_output_file_types_for_task(task_type: TaskType) -> list[str]:
    """
    Get expected output file types for a task type.

    Args:
        task_type: The task type enum.

    Returns:
        List of expected output file types as strings.
    """
    if task_type == TaskType.GENERATE_XML_FROM_INPUT:
        return [FileType.MUSIC_XML.value]
    elif task_type == TaskType.GENERATE_VOICES_FROM_XML:
        return [FileType.AUDIO.value]
    else:
        return []
