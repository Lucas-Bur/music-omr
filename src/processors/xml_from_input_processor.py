"""Processor for converting PDF/PNG/JPG/JPEG input files to MusicXML using homr."""

import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import Future
from dataclasses import dataclass

import fitz  # pymupdf for PDF handling
import cv2
import numpy as np
import onnxruntime as ort
import musicxml.xmlelement.xmlelement as mxl
from musicxml.parser.parser import parse_musicxml

from src.processor import BaseProcessor, ProcessingResult
from src.s3_client import download_s3_object, upload_bytes_to_s3
from src.task_types import FileType, get_input_file_types_for_task, TaskType

from homr import color_adjust
from homr.autocrop import autocrop
from homr.bar_line_detection import detect_bar_lines, prepare_bar_line_image
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug
from homr.model import InputPredictions, MultiStaff
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.segmentation.inference_segnet import extract
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import parse_staffs
from homr.title_detection import detect_title
from homr.transformer.configs import Config
from homr.type_definitions import NDArray


@dataclass
class PredictedSymbols:
    """Container for predicted musical symbols."""

    noteheads: list[BoundingEllipse]
    staff_fragments: list[RotatedBoundingBox]
    clefs_keys: list[RotatedBoundingBox]
    stems_rest: list[RotatedBoundingBox]
    bar_lines: list[RotatedBoundingBox]


@dataclass
class ProcessingConfig:
    """Configuration for processing images."""

    enable_debug: bool = False
    enable_cache: bool = False
    write_staff_positions: bool = False
    read_staff_positions: bool = False
    selected_staff: int = -1
    use_gpu_inference: bool = False


class XMLFromInputProcessor(BaseProcessor):
    """
    Processor that converts PDF/PNG/JPG/JPEG input files to MusicXML using homr.

    This processor handles the 'generate_xml_from_input' task type.
    """

    def __init__(self, s3_client: Any, bucket: str):
        """
        Initialize the XML from input processor.

        Args:
            s3_client: boto3 S3 client instance.
            bucket: S3 bucket for file operations.
        """
        self.s3_client = s3_client
        self.bucket = bucket

    def get_name(self) -> str:
        return "xml_from_input_processor"

    @property
    def supported_input_types(self) -> List[str]:
        return get_input_file_types_for_task(TaskType.GENERATE_XML_FROM_INPUT)

    def process(
        self,
        job_id: str,
        inputs: Dict[str, Any],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> ProcessingResult:
        """
        Process a PDF/PNG/JPG/JPEG file and convert to MusicXML.

        Args:
            job_id: The job/song identifier.
            inputs: Dict containing 'input_key' for the input file.
            progress_callback: Optional callback to report progress (0-100).

        Returns:
            ProcessingResult with MusicXML output file info.
        """
        input_key = inputs.get("input_key")
        if not input_key:
            return ProcessingResult(
                success=False,
                output_files=[],
                error_message="Missing input_key in inputs",
            )

        print(f"Debug: processing job {job_id} with input_key {input_key}")

        # Use context manager for automatic cleanup of temp directory
        with tempfile.TemporaryDirectory(prefix=f"omr_{job_id}_") as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Download input file using existing S3 client function
                input_bytes = download_s3_object(self.s3_client, self.bucket, input_key)

                # Determine file type and process accordingly
                file_extension = input_key.lower().split(".")[-1]

                if file_extension == "pdf":
                    # Convert PDF to images
                    image_paths = self._convert_pdf_to_images(input_bytes, temp_path)
                elif file_extension in ["png", "jpg", "jpeg"]:
                    # Save image to temp file
                    image_path = temp_path / f"input.{file_extension}"
                    image_path.write_bytes(input_bytes)
                    image_paths = [image_path]
                else:
                    return ProcessingResult(
                        success=False,
                        output_files=[],
                        error_message=f"Unsupported file type: {file_extension}",
                    )

                # Configure processing
                has_gpu_support = (
                    "CUDAExecutionProvider" in ort.get_available_providers()
                )
                config = ProcessingConfig(
                    enable_debug=False,
                    enable_cache=False,
                    write_staff_positions=False,
                    read_staff_positions=False,
                    selected_staff=-1,
                    use_gpu_inference=has_gpu_support,
                )

                xml_generator_args = XmlGeneratorArguments(
                    large_page=False,
                    metronome=None,
                    tempo=None,
                )

                # Process each page/image and save to temp files
                xml_files = []
                for idx, image_path in enumerate(image_paths):
                    print(f"Processing page {idx + 1}/{len(image_paths)}")
                    if progress_callback:
                        progress = int(((idx) / len(image_paths)) * 100)
                        progress_callback(progress)
                    xml_content = self._process_single_image(
                        str(image_path), config, xml_generator_args
                    )

                    # Save XML to temp file for merging
                    xml_file_path = temp_path / f"page_{idx:04d}.musicxml"
                    xml_file_path.write_text(xml_content)
                    xml_files.append(xml_file_path)

                # Merge multiple pages into single MusicXML
                if len(xml_files) > 1:
                    print(f"Merging {len(xml_files)} pages into single MusicXML")
                    final_xml = self._merge_musicxml_files(xml_files, temp_path)
                else:
                    final_xml = xml_files[0].read_text()

                if progress_callback:
                    progress_callback(100)

                # Upload output using existing S3 client function
                output_key = f"jobs/{job_id}/music_{int(time.time())}.xml"
                upload_bytes_to_s3(
                    self.s3_client,
                    self.bucket,
                    output_key,
                    final_xml.encode("utf-8"),
                    "application/vnd.recordare.musicxml+xml",
                )

                return ProcessingResult(
                    success=True,
                    output_files=[
                        {
                            "file_type": FileType.MUSIC_XML.value,
                            "s3_key": output_key,
                            "size_bytes": len(final_xml.encode("utf-8")),
                        }
                    ],
                    error_message=None,
                )

            except Exception as e:
                print(f"Error processing file: {e}")
                return ProcessingResult(
                    success=False,
                    output_files=[],
                    error_message=str(e),
                )

    def _convert_pdf_to_images(self, pdf_bytes: bytes, temp_dir: Path) -> list[Path]:
        """
        Convert PDF bytes to list of image file paths.

        Args:
            pdf_bytes: PDF file content as bytes.
            temp_dir: Temporary directory for storing images.

        Returns:
            List of paths to converted images.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        image_paths = []

        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # 2x zoom for better quality
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_path = temp_dir / f"page_{page_num:04d}.png"
                pix.save(image_path)
                image_paths.append(image_path)
        finally:
            doc.close()

        return image_paths

    def _get_predictions(
        self,
        original: NDArray,
        preprocessed: NDArray,
        img_path: str,
        enable_cache: bool,
        use_gpu_inference: bool,
    ) -> InputPredictions:
        """Extract predictions using homr's segmentation model."""
        result = extract(
            preprocessed,
            img_path,
            step_size=320,
            use_cache=enable_cache,
            use_gpu_inference=use_gpu_inference,
        )
        original_image = cv2.resize(
            original, (result.staff.shape[1], result.staff.shape[0])
        )
        preprocessed_image = cv2.resize(
            preprocessed, (result.staff.shape[1], result.staff.shape[0])
        )
        return InputPredictions(
            original=original_image,
            preprocessed=preprocessed_image,
            notehead=result.notehead.astype(np.uint8),
            symbols=result.symbols.astype(np.uint8),
            staff=result.staff.astype(np.uint8),
            clefs_keys=result.clefs_keys.astype(np.uint8),
            stems_rest=result.stems_rests.astype(np.uint8),
        )

    def _load_and_preprocess_predictions(
        self,
        image_path: str,
        enable_debug: bool,
        enable_cache: bool,
        use_gpu_inference: bool,
    ) -> tuple[InputPredictions, Debug]:
        """Load and preprocess image for prediction."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = autocrop(image)
        image = resize_image(image)
        preprocessed, _background = color_adjust.color_adjust(image, 40)
        predictions = self._get_predictions(
            image, preprocessed, image_path, enable_cache, use_gpu_inference
        )
        debug = Debug(predictions.original, image_path, enable_debug)

        predictions = filter_predictions(predictions, debug)
        predictions.staff = make_lines_stronger(predictions.staff, (1, 2))

        return predictions, debug

    def _predict_symbols(
        self, debug: Debug, predictions: InputPredictions
    ) -> PredictedSymbols:
        """Create bounding boxes for all musical symbols."""
        print("Creating bounds for noteheads")
        noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))

        print("Creating bounds for staff_fragments")
        staff_fragments = create_rotated_bounding_boxes(
            predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
        )

        print("Creating bounds for clefs_keys")
        clefs_keys = create_rotated_bounding_boxes(
            predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
        )

        print("Creating bounds for stems_rest")
        stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)

        print("Creating bounds for bar_lines")
        bar_line_img = prepare_bar_line_image(predictions.stems_rest)
        bar_lines = create_rotated_bounding_boxes(
            bar_line_img, skip_merging=True, min_size=(1, 5)
        )

        return PredictedSymbols(
            noteheads, staff_fragments, clefs_keys, stems_rest, bar_lines
        )

    def _detect_staffs_in_image(
        self,
        image_path: str,
        config: ProcessingConfig,
    ) -> tuple[list[MultiStaff], NDArray, Debug, Future]:
        """Detect staffs and parse musical content."""
        predictions, debug = self._load_and_preprocess_predictions(
            image_path,
            config.enable_debug,
            config.enable_cache,
            config.use_gpu_inference,
        )
        symbols = self._predict_symbols(debug, predictions)
        print(
            f"Debug: predicted symbols - noteheads: {len(symbols.noteheads)}, staff_fragments: {len(symbols.staff_fragments)}, clefs_keys: {len(symbols.clefs_keys)}, stems_rest: {len(symbols.stems_rest)}, bar_lines: {len(symbols.bar_lines)}"
        )

        symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
        print(f"Found {len(symbols.staff_fragments)} staff line fragments")

        noteheads_with_stems = combine_noteheads_with_stems(
            symbols.noteheads, symbols.stems_rest
        )
        print(f"Found {len(noteheads_with_stems)} noteheads")

        if len(noteheads_with_stems) == 0:
            raise Exception("No noteheads found")

        average_note_head_height = float(
            np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
        )
        print(f"Average note head height: {average_note_head_height}")

        all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
        all_stems = [
            note.stem for note in noteheads_with_stems if note.stem is not None
        ]
        bar_lines_or_rests = [
            line
            for line in symbols.bar_lines
            if not line.is_overlapping_with_any(all_noteheads)
            and not line.is_overlapping_with_any(all_stems)
        ]
        bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
        print(f"Found {len(bar_line_boxes)} bar lines")

        staffs = detect_staff(
            debug,
            predictions.staff,
            symbols.staff_fragments,
            symbols.clefs_keys,
            bar_line_boxes,
        )
        print(f"Debug: detected {len(staffs)} staffs")
        if len(staffs) == 0:
            raise Exception("No staffs found")

        title_future = detect_title(debug, staffs[0])

        brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
        brace_dot = create_rotated_bounding_boxes(
            brace_dot_img, skip_merging=True, max_size=(100, -1)
        )

        notes = add_notes_to_staffs(
            staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
        )

        multi_staffs = find_braces_brackets_and_grand_staff_lines(
            debug, staffs, brace_dot
        )
        print(f"Debug: found {len(multi_staffs)} multi-staffs")
        print(
            f"Found {len(multi_staffs)} connected staffs: {[len(staff.staffs) for staff in multi_staffs]}"
        )

        return multi_staffs, predictions.preprocessed, debug, title_future

    def _process_single_image(
        self,
        image_path: str,
        config: ProcessingConfig,
        xml_generator_args: XmlGeneratorArguments,
    ) -> str:
        """
        Process a single image and return MusicXML content.

        Args:
            image_path: Path to the image file.
            config: Processing configuration.
            xml_generator_args: XML generator arguments.

        Returns:
            MusicXML content as string.
        """
        print(f"Debug: processing image {image_path}")
        print(f"Processing {image_path}")

        multi_staffs, image, debug, title_future = self._detect_staffs_in_image(
            image_path, config
        )

        transformer_config = Config()
        transformer_config.use_gpu_inference = config.use_gpu_inference

        result_staffs = parse_staffs(
            debug,
            multi_staffs,
            image,
            selected_staff=config.selected_staff,
            config=transformer_config,
        )

        title = title_future.result(60)
        print(f"Found title: {title}")

        print("Generating XML")
        xml = generate_xml(xml_generator_args, result_staffs, title)

        print(f"Finished parsing {len(result_staffs)} staves")

        return xml.to_string()

    def _clef_attributes(self, clef: mxl.XMLClef) -> Dict[str, Any]:
        """Extract clef attributes from XMLClef element."""
        attributes = {}
        for a in clef.get_children_of_type(mxl.XMLSign):
            attributes["Sign"] = a.value_  # type: ignore
        for a in clef.get_children_of_type(mxl.XMLLine):
            attributes["Line"] = a.value_  # type: ignore
        for a in clef.get_children_of_type(mxl.XMLClefOctaveChange):
            attributes["ClefOctaveChange"] = a.value_  # type: ignore
        return attributes

    def _time_attributes(self, time: mxl.XMLTime) -> Dict[str, Any]:
        """Extract time signature attributes from XMLTime element."""
        attributes = {}
        for a in time.get_children_of_type(mxl.XMLBeats):
            attributes["Beats"] = a.value_  # type: ignore
        for a in time.get_children_of_type(mxl.XMLBeatType):
            attributes["BeatType"] = a.value_  # type: ignore
        for a in time.get_children_of_type(mxl.XMLInterchangeable):
            attributes["Interchangeable"] = a.value_  # type: ignore
        for a in time.get_children_of_type(mxl.XMLSenzaMisura):
            attributes["SenzaMisura"] = a.value_  # type: ignore
        return attributes

    def _key_attributes(self, key: mxl.XMLKey) -> Dict[str, Any]:
        """Extract key signature attributes from XMLKey element."""
        attributes = {}
        for a in key.get_children_of_type(mxl.XMLFifths):
            attributes["Fifths"] = a.value_  # type: ignore
        for a in key.get_children_of_type(mxl.XMLKeyAlter):
            attributes["KeyAlter"] = a.value_  # type: ignore
        for a in key.get_children_of_type(mxl.XMLMode):
            attributes["Mode"] = a.value_  # type: ignore
        return attributes

    def _merge_musicxml_files(self, xml_files: List[Path], temp_dir: Path) -> str:
        """
        Merge multiple MusicXML files into a single MusicXML document.

        This function merges multiple pages by:
        1. Using the first file as the main document
        2. Extracting the last attributes (divisions, key, time, clef) from each part
        3. Adding measures from subsequent files to corresponding parts
        4. Removing duplicate attributes from the first measure of each subsequent file
        5. Renumbering measures to maintain sequential order

        Args:
            xml_files: List of paths to MusicXML files to merge (in order).

        Returns:
            Merged MusicXML content as string.
        """
        if not xml_files:
            raise ValueError("No MusicXML files to merge")

        if len(xml_files) == 1:
            # Single file - just read and return
            return xml_files[0].read_text()

        print(f"Debug: merging {len(xml_files)} MusicXML files")
        # Main file is the first of the list
        main_file = xml_files[0]
        print(f"Starting with {main_file}")
        m = parse_musicxml(str(main_file))

        # Extract the last attributes from each part
        last_parts_attributes = []
        for part in m.get_children_of_type(mxl.XMLPart):
            part_attributes = {}
            for measure in part.get_children_of_type(mxl.XMLMeasure):
                for attrib in measure.get_children_of_type(mxl.XMLAttributes):
                    for div in attrib.get_children_of_type(mxl.XMLDivisions):
                        part_attributes["Divisions"] = div.value_
                    for key in attrib.get_children_of_type(mxl.XMLKey):
                        part_attributes["Key"] = self._key_attributes(key)
                    for xtime in attrib.get_children_of_type(mxl.XMLTime):
                        part_attributes["Time"] = self._time_attributes(xtime)
                    for clef in attrib.get_children_of_type(mxl.XMLClef):
                        part_attributes["Clef"] = self._clef_attributes(clef)
            last_parts_attributes.append(part_attributes)

        # Process subsequent files
        for f in xml_files[1:]:
            print(f"Merging {f}")
            b = parse_musicxml(str(f))
            ip = 0  # Index for main file parts

            for part1 in m.get_children_of_type(mxl.XMLPart):
                # Each part from the main score
                current_len = len(part1.get_children_of_type(mxl.XMLMeasure))
                ib = 0  # Index for new file parts

                for part in b.get_children_of_type(mxl.XMLPart):
                    if ib == ip:
                        # Add the part from the new file having the same order as the part from the main score
                        for measure in part.get_children_of_type(mxl.XMLMeasure):
                            new_number = str(int(measure.number) + current_len)

                            # Remove duplicate attributes from first measure
                            if int(measure.number) == 1:
                                for attrib in measure.get_children_of_type(
                                    mxl.XMLAttributes
                                ):
                                    for div in attrib.get_children_of_type(
                                        mxl.XMLDivisions
                                    ):
                                        if (
                                            last_parts_attributes[ib]["Divisions"]
                                            == div.value_
                                        ):
                                            attrib.remove(div)
                                            print(
                                                f"Remove division at measure {new_number}, part {ib + 1}"
                                            )

                                    for key in attrib.get_children_of_type(mxl.XMLKey):
                                        if last_parts_attributes[ib][
                                            "Key"
                                        ] == self._key_attributes(key):
                                            attrib.remove(key)
                                            print(
                                                f"Remove key at measure {new_number}, part {ib + 1}"
                                            )

                                    for xtime in attrib.get_children_of_type(
                                        mxl.XMLTime
                                    ):
                                        if last_parts_attributes[ib].get(
                                            "Time"
                                        ) == self._time_attributes(xtime):
                                            attrib.remove(xtime)
                                            print(
                                                f"Remove time at measure {new_number}, part {ib + 1}"
                                            )

                                    for clef in attrib.get_children_of_type(
                                        mxl.XMLClef
                                    ):
                                        if last_parts_attributes[ib][
                                            "Clef"
                                        ] == self._clef_attributes(clef):
                                            attrib.remove(clef)
                                            print(
                                                f"Remove clef at measure {new_number}, part {ib + 1}"
                                            )

                            measure.number = new_number
                            part1.add_child(measure)
                            print(f"Added measure {new_number}, part {ib + 1}")

                        current_len = len(part1.get_children_of_type(mxl.XMLMeasure))
                    ib += 1
                ip += 1

        # Write merged MusicXML to string
        merged_file = temp_dir / "merged.musicxml"
        m.write(str(merged_file))
        return merged_file.read_text()
