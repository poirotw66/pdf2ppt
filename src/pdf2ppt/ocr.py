from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
from PIL import Image

from .core import OcrInitializationError, OcrPageData, OcrProcessingError
from .models import TextBlock

logger = logging.getLogger(__name__)


class OcrEngine:
    def __init__(
        self,
        lang: str,
        *,
        use_doc_unwarping: bool,
        det_thresh: float | None,
        det_box_thresh: float | None,
        drop_score: float | None,
    ) -> None:
        self.lang = lang
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

            engine_kwargs: dict[str, Any] = {
                "lang": self.lang,
                "ocr_version": "PP-OCRv5",
                "use_doc_orientation_classify": True,
                "use_doc_unwarping": self.use_doc_unwarping,
            }
            if self.det_thresh is not None:
                engine_kwargs["text_det_thresh"] = self.det_thresh
            if self.det_box_thresh is not None:
                engine_kwargs["text_det_box_thresh"] = self.det_box_thresh
            if self.drop_score is not None:
                engine_kwargs["text_rec_score_thresh"] = self.drop_score

            logger.info("Initializing PaddleOCR engine for lang=%s", self.lang)
            try:
                self._engine = PaddleOCR(**engine_kwargs)
            except Exception as error:
                raise OcrInitializationError(
                    "Failed to initialize PaddleOCR. Verify model downloads and environment dependencies."
                ) from error
        return self._engine

    def extract_text_blocks(self, image: Image.Image, page_number: int) -> OcrPageData:
        image_array = np.array(image.convert("RGB"))
        try:
            results = self._get_engine().predict(image_array)
        except OcrInitializationError:
            raise
        except Exception as error:
            raise OcrProcessingError(f"OCR prediction failed for page {page_number}: {error}") from error
        blocks: list[TextBlock] = []
        reference_image = image.convert("RGB")
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
