from __future__ import annotations

import json
import logging
import os
import shutil
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


class OcrEngine:
    def __init__(
        self,
        lang: str,
        *,
        model_root: Path | None,
        use_textline_orientation: bool,
        use_doc_unwarping: bool,
        det_thresh: float | None,
        det_box_thresh: float | None,
        drop_score: float | None,
    ) -> None:
        self.lang = lang
        self.model_root = model_root
        self.use_textline_orientation = use_textline_orientation
        self.use_doc_unwarping = use_doc_unwarping
        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        self.drop_score = drop_score
        self._engine: Any | None = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            try:
                from paddleocr import PaddleOCR
            except ImportError as error:
                raise OcrInitializationError(
                    "Failed to import PaddleOCR. Check that the OCR environment is installed and compatible."
                ) from error

            engine_kwargs = self._build_engine_kwargs()
            logger.info(
                "Initializing PaddleOCR engine for lang=%s with model_root=%s textline_orientation=%s",
                self.lang,
                self.model_root,
                self.use_textline_orientation,
            )
            try:
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
            use_doc_unwarping=self.use_doc_unwarping,
            use_textline_orientation=self.use_textline_orientation,
        )
        engine_kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": True,
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
        logger.info("OCR extracted %s block(s) from page %s", len(blocks), page_number)
        return OcrPageData(blocks=blocks, image=reference_image)


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


def normalize_polygon(polygon: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in polygon)


def polygon_to_bbox(polygon: Any) -> tuple[float, float, float, float]:
    points = list(polygon)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def build_local_ocr_model_kwargs(
    *,
    model_root: Path | None,
    lang: str,
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
