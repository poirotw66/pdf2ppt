from __future__ import annotations

import json
import logging
import os
import re
import shutil
import statistics
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import OcrInitializationError, OcrPageData, OcrProcessingError
from .models import TextBlock

logger = logging.getLogger(__name__)


DEFAULT_PPOCR_VERSION = "PP-OCRv5"
DEFAULT_DOC_ORIENTATION_MODEL_NAME = "PP-LCNet_x1_0_doc_ori"
DEFAULT_TEXTLINE_ORIENTATION_MODEL_NAME = "PP-LCNet_x0_25_textline_ori"
DEFAULT_DOC_UNWARPING_MODEL_NAME = "UVDoc"
DEFAULT_TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
DEFAULT_SERVER_REC_LANGS = {"ch", "chinese_cht", "japan"}
DEFAULT_MOBILE_REC_MODELS = {
    "en": "en_PP-OCRv5_mobile_rec",
    "korean": "korean_PP-OCRv5_mobile_rec",
    "th": "th_PP-OCRv5_mobile_rec",
    "el": "el_PP-OCRv5_mobile_rec",
    "te": "te_PP-OCRv5_mobile_rec",
    "ta": "ta_PP-OCRv5_mobile_rec",
}
DEFAULT_GROUPED_REC_MODEL_PREFIXES = {
    "latin": {
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
    },
    "eslav": {"be", "bg", "uk", "mn", "abq", "ady", "kbd", "ava", "dar", "inh", "che", "lbe", "lez", "tab"},
    "arabic": {"ar", "fa", "ug", "ur"},
    "cyrillic": {"ru", "rs_cyrillic"},
    "devanagari": {"hi", "mr", "ne"},
}
PADDLEX_OFFICIAL_MODEL_CACHE_DIR = Path.home() / ".paddlex" / "official_models"
DEFAULT_OCR_LINE_MERGE_GAP_HEIGHT_RATIO = 1.25
DEFAULT_OCR_WORD_GAP_HEIGHT_RATIO = 0.35
DEFAULT_OCR_LINE_VERTICAL_OVERLAP_RATIO = 0.3
DEFAULT_OCR_LINE_CENTER_GAP_HEIGHT_RATIO = 0.75
DEFAULT_OCR_MINOR_FRAGMENT_HEIGHT_RATIO = 0.25
DEFAULT_OCR_MINOR_FRAGMENT_WIDTH_RATIO = 0.4
_CJK_CHAR_PATTERN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


@contextmanager
def suppress_known_paddle_runtime_warnings() -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"No ccache found\..*",
            category=UserWarning,
        )
        yield


class OcrEngine:
    def __init__(
        self,
        lang: str,
        *,
        model_root: Path | None,
        use_doc_orientation: bool,
        use_textline_orientation: bool,
        use_doc_unwarping: bool,
        det_thresh: float | None,
        det_box_thresh: float | None,
        drop_score: float | None,
        return_word_box: bool = False,
    ) -> None:
        self.lang = lang
        self.model_root = model_root
        self.use_doc_orientation = use_doc_orientation
        self.use_textline_orientation = use_textline_orientation
        self.use_doc_unwarping = use_doc_unwarping
        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        self.drop_score = drop_score
        self.return_word_box = return_word_box
        self._engine: Any | None = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            try:
                with suppress_known_paddle_runtime_warnings():
                    from paddleocr import PaddleOCR
            except ImportError as error:
                raise OcrInitializationError(
                    "Failed to import PaddleOCR. Check that the OCR environment is installed and compatible."
                ) from error

            engine_kwargs = self._build_engine_kwargs()
            logger.info(
                "Initializing PaddleOCR engine for lang=%s with model_root=%s doc_orientation=%s textline_orientation=%s",
                self.lang,
                self.model_root,
                self.use_doc_orientation,
                self.use_textline_orientation,
            )
            try:
                with suppress_known_paddle_runtime_warnings():
                    self._engine = PaddleOCR(**engine_kwargs)
            except Exception as error:
                raise OcrInitializationError(
                    "Failed to initialize PaddleOCR. Verify model downloads and environment dependencies."
                ) from error
        return self._engine

    def _build_engine_kwargs(self) -> dict[str, Any]:
        local_model_kwargs = build_local_ocr_model_kwargs(
            model_root=self.model_root,
            lang=self.lang,
            use_doc_orientation=self.use_doc_orientation,
            use_doc_unwarping=self.use_doc_unwarping,
            use_textline_orientation=self.use_textline_orientation,
        )
        engine_kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": self.use_doc_orientation,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
        }
        if local_model_kwargs:
            engine_kwargs.update(local_model_kwargs)
        else:
            engine_kwargs["lang"] = self.lang
            engine_kwargs["ocr_version"] = DEFAULT_PPOCR_VERSION
        if self.det_thresh is not None:
            engine_kwargs["text_det_thresh"] = self.det_thresh
        if self.det_box_thresh is not None:
            engine_kwargs["text_det_box_thresh"] = self.det_box_thresh
        if self.drop_score is not None:
            engine_kwargs["text_rec_score_thresh"] = self.drop_score
        engine_kwargs["return_word_box"] = self.return_word_box
        return engine_kwargs

    def extract_text_blocks(self, image: Image.Image, page_number: int) -> OcrPageData:
        rgb_image = image.convert("RGB")
        image_array = np.array(rgb_image)
        try:
            results = self._get_engine().predict(image_array)
        except OcrInitializationError:
            raise
        except Exception as error:
            raise OcrProcessingError(f"OCR prediction failed for page {page_number}: {error}") from error
        blocks: list[TextBlock] = []
        reference_image = rgb_image
        order = 1
        for result in results:
            payload = _coerce_ocr_payload(result)
            candidate_image = _extract_ocr_reference_image(payload)
            if candidate_image is not None:
                reference_image = candidate_image
            extracted = _extract_ocr_blocks(payload, page_number=page_number, order_start=order)
            blocks.extend(extracted)
            order += len(extracted)
        raw_count = len(blocks)
        blocks = merge_adjacent_ocr_line_blocks(blocks, page_number=page_number)
        logger.info(
            "OCR extracted %s block(s) from page %s (%s before line merge)",
            len(blocks),
            page_number,
            raw_count,
        )
        return OcrPageData(blocks=blocks, image=reference_image)

    def extract_text_blocks_batch(self, images: list[Image.Image], page_numbers: list[int]) -> list[OcrPageData]:
        if len(images) != len(page_numbers):
            raise ValueError("images and page_numbers must have the same length")
        if not images:
            return []

        rgb_images = [image.convert("RGB") for image in images]
        image_arrays = [np.array(image) for image in rgb_images]
        try:
            results = list(self._get_engine().predict(image_arrays))
        except OcrInitializationError:
            raise
        except Exception as error:
            raise OcrProcessingError(
                f"OCR batch prediction failed for pages {page_numbers[0]}-{page_numbers[-1]}: {error}"
            ) from error

        if len(results) != len(images):
            return [self.extract_text_blocks(image, page_number) for image, page_number in zip(images, page_numbers)]

        page_data_list: list[OcrPageData] = []
        for rgb_image, page_number, result in zip(rgb_images, page_numbers, results):
            blocks: list[TextBlock] = []
            reference_image = rgb_image
            order = 1
            payload = _coerce_ocr_payload(result)
            candidate_image = _extract_ocr_reference_image(payload)
            if candidate_image is not None:
                reference_image = candidate_image
            extracted = _extract_ocr_blocks(payload, page_number=page_number, order_start=order)
            blocks.extend(extracted)
            raw_count = len(blocks)
            blocks = merge_adjacent_ocr_line_blocks(blocks, page_number=page_number)
            logger.info(
                "OCR batch extracted %s block(s) from page %s (%s before line merge)",
                len(blocks),
                page_number,
                raw_count,
            )
            page_data_list.append(OcrPageData(blocks=blocks, image=reference_image))
        return page_data_list

    def recognize_text_in_box(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
        *,
        page_number: int,
    ) -> tuple[str, float]:
        crop_box = clamp_bbox_to_image(image.size, bbox)
        if crop_box is None:
            return "", 0.0
        crop = image.convert("RGB").crop(crop_box)
        page_data = self.extract_text_blocks(crop, page_number)
        texts = [block.text.strip() for block in page_data.blocks if block.text.strip()]
        if not texts:
            return "", 0.0
        confidences = [block.confidence for block in page_data.blocks if block.text.strip()]
        confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        return "\n".join(texts), confidence


def _coerce_ocr_payload(result: Any) -> Any:
    if isinstance(result, (dict, list)):
        return result
    if hasattr(result, "res"):
        return result.res
    if hasattr(result, "json"):
        payload = result.json
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            return json.loads(payload)
        return payload
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


def _extract_ocr_reference_image(payload: Any) -> Image.Image | None:
    doc_preprocessor = payload.get("doc_preprocessor_res") if isinstance(payload, dict) else None
    if not isinstance(doc_preprocessor, dict):
        return None
    output_img = doc_preprocessor.get("output_img")
    if isinstance(output_img, np.ndarray):
        return Image.fromarray(output_img).convert("RGB")
    return None


def _extract_ocr_blocks(payload: Any, *, page_number: int, order_start: int) -> list[TextBlock]:
    if isinstance(payload, dict):
        if {"rec_texts", "rec_scores", "dt_polys"}.issubset(payload.keys()):
            return _blocks_from_recognized_payload(
                payload["dt_polys"],
                payload["rec_texts"],
                payload["rec_scores"],
                page_number=page_number,
                order_start=order_start,
            )
        for nested_key in ("ocr_result", "result", "res", "data"):
            if nested_key in payload:
                return _extract_ocr_blocks(payload[nested_key], page_number=page_number, order_start=order_start)

    if isinstance(payload, list):
        if payload and isinstance(payload[0], list) and len(payload[0]) >= 2:
            blocks: list[TextBlock] = []
            for index, item in enumerate(payload, start=order_start):
                polygon = item[0]
                text_payload = item[1]
                if not isinstance(text_payload, (list, tuple)) or len(text_payload) < 2:
                    continue
                text = str(text_payload[0]).strip()
                if not text:
                    continue
                score = float(text_payload[1])
                blocks.append(
                    TextBlock(
                        id=f"ocr_{page_number}_{index}",
                        source="ocr",
                        bbox=polygon_to_bbox(polygon),
                        text=text,
                        confidence=score,
                        reading_order=index,
                        image_bbox=polygon_to_bbox(polygon),
                        image_polygon=normalize_polygon(polygon),
                    )
                )
            return blocks

        blocks: list[TextBlock] = []
        next_order = order_start
        for item in payload:
            extracted = _extract_ocr_blocks(item, page_number=page_number, order_start=next_order)
            blocks.extend(extracted)
            next_order += len(extracted)
        return blocks

    return []


def _blocks_from_recognized_payload(
    polygons: list[Any],
    texts: list[Any],
    scores: list[Any],
    *,
    page_number: int,
    order_start: int,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for index, (polygon, text, score) in enumerate(zip(polygons, texts, scores), start=order_start):
        normalized_text = str(text).strip()
        if not normalized_text:
            continue
        blocks.append(
            TextBlock(
                id=f"ocr_{page_number}_{index}",
                source="ocr",
                bbox=polygon_to_bbox(polygon),
                text=normalized_text,
                confidence=float(score),
                reading_order=index,
                image_bbox=polygon_to_bbox(polygon),
                image_polygon=normalize_polygon(polygon),
            )
        )
    return blocks


def merge_adjacent_ocr_line_blocks(
    blocks: list[TextBlock],
    *,
    page_number: int,
    line_vertical_overlap_ratio: float = DEFAULT_OCR_LINE_VERTICAL_OVERLAP_RATIO,
    line_center_gap_height_ratio: float = DEFAULT_OCR_LINE_CENTER_GAP_HEIGHT_RATIO,
    merge_gap_height_ratio: float = DEFAULT_OCR_LINE_MERGE_GAP_HEIGHT_RATIO,
) -> list[TextBlock]:
    if len(blocks) <= 1:
        return _reindex_ocr_blocks(blocks, page_number=page_number)

    heights = [max(1.0, block.bbox[3] - block.bbox[1]) for block in blocks]
    median_height = statistics.median(heights)
    gap_threshold = median_height * merge_gap_height_ratio
    center_gap_threshold = median_height * line_center_gap_height_ratio

    major_blocks = [block for block in blocks if not _is_minor_ocr_fragment(block, median_height)]
    minor_blocks = [block for block in blocks if _is_minor_ocr_fragment(block, median_height)]
    if not major_blocks:
        major_blocks = list(blocks)
        minor_blocks = []

    line_groups: list[list[TextBlock]] = []
    for block in sorted(major_blocks, key=_ocr_block_sort_key):
        target_line: list[TextBlock] | None = None
        for line in line_groups:
            if any(
                _blocks_share_ocr_line(
                    block,
                    line_block,
                    min_overlap_ratio=line_vertical_overlap_ratio,
                    center_gap_threshold=center_gap_threshold,
                )
                for line_block in line
            ):
                target_line = line
                break
        if target_line is None:
            line_groups.append([block])
        else:
            target_line.append(block)

    for block in minor_blocks:
        target_line = _find_best_ocr_line_group(block, line_groups)
        if target_line is None:
            line_groups.append([block])
        else:
            target_line.append(block)

    merged_blocks: list[TextBlock] = []
    for line in line_groups:
        line.sort(key=lambda block: block.bbox[0])
        current_group = [line[0]]
        for block in line[1:]:
            if _can_merge_horizontally(
                current_group,
                block,
                gap_threshold=gap_threshold,
                word_gap_threshold=median_height * DEFAULT_OCR_WORD_GAP_HEIGHT_RATIO,
                center_gap_threshold=center_gap_threshold,
                min_overlap_ratio=line_vertical_overlap_ratio,
            ):
                current_group.append(block)
                continue
            merged_blocks.append(_merge_ocr_block_group(current_group))
            current_group = [block]
        merged_blocks.append(_merge_ocr_block_group(current_group))

    return _reindex_ocr_blocks(merged_blocks, page_number=page_number)


def _can_merge_horizontally(
    group: list[TextBlock],
    block: TextBlock,
    *,
    gap_threshold: float,
    word_gap_threshold: float,
    center_gap_threshold: float,
    min_overlap_ratio: float,
) -> bool:
    group_extent = _blocks_extent(group)
    gap = block.bbox[0] - group_extent[2]
    if gap > gap_threshold:
        return False
    group_block = TextBlock(
        id="merge-group",
        source="ocr",
        bbox=group_extent,
        text="",
        confidence=1.0,
    )
    if not _blocks_share_ocr_line(
        group_block,
        block,
        min_overlap_ratio=min_overlap_ratio,
        center_gap_threshold=center_gap_threshold,
    ):
        return False
    if gap <= 0 or gap <= word_gap_threshold:
        return True
    return _horizontal_bbox_overlap(group_extent, block.bbox)


def _horizontal_bbox_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return left[2] > right[0] and right[2] > left[0]


def _blocks_extent(blocks: list[TextBlock]) -> tuple[float, float, float, float]:
    return (
        min(block.bbox[0] for block in blocks),
        min(block.bbox[1] for block in blocks),
        max(block.bbox[2] for block in blocks),
        max(block.bbox[3] for block in blocks),
    )


def _is_minor_ocr_fragment(block: TextBlock, median_height: float) -> bool:
    height = max(1.0, block.bbox[3] - block.bbox[1])
    width = max(1.0, block.bbox[2] - block.bbox[0])
    if height < median_height * DEFAULT_OCR_MINOR_FRAGMENT_HEIGHT_RATIO:
        return True
    return (
        height < median_height * DEFAULT_OCR_MINOR_FRAGMENT_WIDTH_RATIO
        and width < median_height * DEFAULT_OCR_MINOR_FRAGMENT_WIDTH_RATIO
    )


def _find_best_ocr_line_group(
    block: TextBlock,
    line_groups: list[list[TextBlock]],
) -> list[TextBlock] | None:
    best_line: list[TextBlock] | None = None
    best_overlap = 0.0
    for line in line_groups:
        overlap = max(_vertical_overlap_height(block, line_block) for line_block in line)
        if overlap > best_overlap:
            best_overlap = overlap
            best_line = line
    return best_line


def _vertical_overlap_height(left: TextBlock, right: TextBlock) -> float:
    overlap_top = max(left.bbox[1], right.bbox[1])
    overlap_bottom = min(left.bbox[3], right.bbox[3])
    return max(0.0, overlap_bottom - overlap_top)


def _blocks_share_ocr_line(
    left: TextBlock,
    right: TextBlock,
    *,
    min_overlap_ratio: float,
    center_gap_threshold: float,
) -> bool:
    left_center_y = (left.bbox[1] + left.bbox[3]) / 2.0
    right_center_y = (right.bbox[1] + right.bbox[3]) / 2.0
    if abs(left_center_y - right_center_y) > center_gap_threshold:
        return False
    overlap_height = _vertical_overlap_height(left, right)
    if overlap_height <= 0:
        return False
    min_height = min(left.bbox[3] - left.bbox[1], right.bbox[3] - right.bbox[1], 1.0)
    return overlap_height / min_height >= min_overlap_ratio


def _ocr_block_sort_key(block: TextBlock) -> tuple[float, float]:
    return ((block.bbox[1] + block.bbox[3]) / 2.0, block.bbox[0])


def _reindex_ocr_blocks(blocks: list[TextBlock], *, page_number: int) -> list[TextBlock]:
    reindexed: list[TextBlock] = []
    for order, block in enumerate(blocks, start=1):
        reindexed.append(
            TextBlock(
                id=f"ocr_{page_number}_{order}",
                source=block.source,
                bbox=block.bbox,
                text=block.text,
                confidence=block.confidence,
                font_family=block.font_family,
                font_size=block.font_size,
                font_color=block.font_color,
                bold=block.bold,
                italic=block.italic,
                reading_order=order,
                block_role=block.block_role,
                image_bbox=block.image_bbox,
                image_polygon=block.image_polygon,
            )
        )
    return reindexed


def _merge_ocr_block_group(blocks: list[TextBlock]) -> TextBlock:
    anchor = blocks[0]
    merged_bbox = (
        min(block.bbox[0] for block in blocks),
        min(block.bbox[1] for block in blocks),
        max(block.bbox[2] for block in blocks),
        max(block.bbox[3] for block in blocks),
    )
    image_bboxes = [block.image_bbox for block in blocks if block.image_bbox is not None]
    merged_image_bbox = (
        (
            min(bbox[0] for bbox in image_bboxes),
            min(bbox[1] for bbox in image_bboxes),
            max(bbox[2] for bbox in image_bboxes),
            max(bbox[3] for bbox in image_bboxes),
        )
        if image_bboxes
        else None
    )
    polygon_points: list[tuple[float, float]] = []
    for block in blocks:
        if block.image_polygon:
            polygon_points.extend(block.image_polygon)
    merged_polygon = (
        (
            (merged_image_bbox[0], merged_image_bbox[1]),
            (merged_image_bbox[2], merged_image_bbox[1]),
            (merged_image_bbox[2], merged_image_bbox[3]),
            (merged_image_bbox[0], merged_image_bbox[3]),
        )
        if merged_image_bbox is not None
        else normalize_polygon(polygon_points) if polygon_points else None
    )
    merged_text = _join_ocr_text_segments(block.text for block in blocks)
    total_chars = sum(len(block.text) for block in blocks) or 1
    merged_confidence = sum(block.confidence * len(block.text) for block in blocks) / total_chars
    return TextBlock(
        id=anchor.id,
        source=anchor.source,
        bbox=merged_bbox,
        text=merged_text,
        confidence=merged_confidence,
        font_family=anchor.font_family,
        font_size=anchor.font_size,
        font_color=anchor.font_color,
        bold=anchor.bold,
        italic=anchor.italic,
        reading_order=anchor.reading_order,
        block_role=anchor.block_role,
        image_bbox=merged_image_bbox,
        image_polygon=merged_polygon,
    )


def _join_ocr_text_segments(texts: list[str]) -> str:
    segments = [text.strip() for text in texts if text.strip()]
    if not segments:
        return ""
    merged = segments[0]
    for segment in segments[1:]:
        if _needs_space_between(merged, segment):
            merged = f"{merged} {segment}"
        else:
            merged = f"{merged}{segment}"
    return merged


def _needs_space_between(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left[-1].isspace() or right[0].isspace():
        return False
    if _CJK_CHAR_PATTERN.search(left[-1]) or _CJK_CHAR_PATTERN.search(right[0]):
        return False
    return left[-1].isalnum() and right[0].isalnum()


def normalize_polygon(polygon: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in polygon)


def polygon_to_bbox(polygon: Any) -> tuple[float, float, float, float]:
    points = list(polygon)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def clamp_bbox_to_image(
    image_size: tuple[int, int],
    bbox: tuple[float, float, float, float],
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    left = max(0, min(width, int(bbox[0])))
    top = max(0, min(height, int(bbox[1])))
    right = max(left, min(width, int(bbox[2])))
    bottom = max(top, min(height, int(bbox[3])))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_local_ocr_model_kwargs(
    *,
    model_root: Path | None,
    lang: str,
    use_doc_orientation: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
) -> dict[str, str]:
    if model_root is None:
        return {}

    recognition_model_name = resolve_recognition_model_name(lang)
    if recognition_model_name is None:
        logger.warning(
            "No explicit local PaddleOCR recognition model mapping for lang=%s; falling back to PaddleOCR defaults.",
            lang,
        )
        return {}

    resolved_root = model_root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    model_kwargs: dict[str, str] = {}
    if use_doc_orientation:
        model_kwargs.update(
            _build_model_location(
                model_root=resolved_root,
                name_key="doc_orientation_classify_model_name",
                dir_key="doc_orientation_classify_model_dir",
                model_name=DEFAULT_DOC_ORIENTATION_MODEL_NAME,
            )
        )
    model_kwargs.update(
        _build_model_location(
            model_root=resolved_root,
            name_key="text_detection_model_name",
            dir_key="text_detection_model_dir",
            model_name=DEFAULT_TEXT_DETECTION_MODEL_NAME,
        )
    )
    model_kwargs.update(
        _build_model_location(
            model_root=resolved_root,
            name_key="text_recognition_model_name",
            dir_key="text_recognition_model_dir",
            model_name=recognition_model_name,
        )
    )
    if use_textline_orientation:
        model_kwargs.update(
            _build_model_location(
                model_root=resolved_root,
                name_key="textline_orientation_model_name",
                dir_key="textline_orientation_model_dir",
                model_name=DEFAULT_TEXTLINE_ORIENTATION_MODEL_NAME,
            )
        )

    if use_doc_unwarping:
        model_kwargs.update(
            _build_model_location(
                model_root=resolved_root,
                name_key="doc_unwarping_model_name",
                dir_key="doc_unwarping_model_dir",
                model_name=DEFAULT_DOC_UNWARPING_MODEL_NAME,
            )
        )

    return model_kwargs


def resolve_recognition_model_name(lang: str) -> str | None:
    if lang in DEFAULT_SERVER_REC_LANGS:
        return "PP-OCRv5_server_rec"
    if lang in DEFAULT_MOBILE_REC_MODELS:
        return DEFAULT_MOBILE_REC_MODELS[lang]
    for group_name, supported_langs in DEFAULT_GROUPED_REC_MODEL_PREFIXES.items():
        if lang in supported_langs:
            return f"{group_name}_PP-OCRv5_mobile_rec"
    return None


def _build_model_location(
    *,
    model_root: Path,
    name_key: str,
    dir_key: str,
    model_name: str,
) -> dict[str, str]:
    local_model_dir = ensure_local_model_dir(model_root, model_name)
    return {name_key: model_name, dir_key: str(local_model_dir)}


def ensure_local_model_dir(model_root: Path, model_name: str) -> Path:
    model_root.mkdir(parents=True, exist_ok=True)
    local_dir = model_root / model_name
    if local_dir.exists():
        return local_dir

    cached_dir = PADDLEX_OFFICIAL_MODEL_CACHE_DIR / model_name
    if cached_dir.exists():
        try:
            local_dir.symlink_to(cached_dir, target_is_directory=True)
            return local_dir
        except OSError:
            shutil.copytree(cached_dir, local_dir)
            return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir
