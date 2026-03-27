from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw

from .core import ConversionOptions
from .debug_artifacts import build_mask_shapes
from .models import TextBlock

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BackgroundRenderResult:
    image: Image.Image
    engine_name: str | None
    note: str | None
    mask_image: Image.Image | None = None


class BackgroundInpaintingError(RuntimeError):
    pass


class BackgroundInpaintingEngine:
    name = "base"

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        raise NotImplementedError


class WhiteBoxInpaintingEngine(BackgroundInpaintingEngine):
    name = "white-box"

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        mask_array = np.array(mask_image.convert("L")) > 0
        result = np.array(page_image.convert("RGB"), copy=True)
        result[mask_array] = 255
        return Image.fromarray(result, mode="RGB")


class OpenCvFastInpaintingEngine(BackgroundInpaintingEngine):
    name = "opencv-fast"

    def __init__(self, *, radius: float = 3.0) -> None:
        self.radius = radius

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
        if np.count_nonzero(mask_array) == 0:
            return page_image.convert("RGB").copy()
        source = cv2.cvtColor(np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        repaired = cv2.inpaint(source, mask_array, self.radius, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))


class DiffusionLocalInpaintingEngine(BackgroundInpaintingEngine):
    name = "diffusion-local"

    def __init__(
        self,
        *,
        command: str,
        model: str,
        device: str,
        max_crop_edge: int,
        crop_padding_px: int,
        timeout_sec: float,
    ) -> None:
        self.command = command
        self.model = model
        self.device = device
        self.max_crop_edge = max(64, max_crop_edge)
        self.crop_padding_px = max(0, crop_padding_px)
        self.timeout_sec = max(1.0, timeout_sec)

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        if shutil.which(self.command) is None:
            raise BackgroundInpaintingError(
                f"Diffusion backend command '{self.command}' was not found in PATH."
            )

        crop_box = compute_mask_crop_box(mask_image, padding_px=self.crop_padding_px)
        if crop_box is None:
            logger.debug("No non-zero mask crop detected; returning original background")
            return page_image.convert("RGB").copy()

        base_image = page_image.convert("RGB")
        crop_image = base_image.crop(crop_box)
        crop_mask = mask_image.convert("L").crop(crop_box)
        processed_crop = invoke_diffusion_backend(
            crop_image,
            crop_mask,
            command=self.command,
            model=self.model,
            device=self.device,
            max_crop_edge=self.max_crop_edge,
            timeout_sec=self.timeout_sec,
        )
        result = base_image.copy()
        result.paste(processed_crop, crop_box)
        return result


def mask_text_regions_with_white_boxes(
    page_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
) -> Image.Image:
    mask_image = build_text_mask_image(text_blocks, page_image.size, page_rect, padding_px=0)
    return WhiteBoxInpaintingEngine().inpaint(page_image, mask_image)


def build_text_mask_image(
    text_blocks: list[TextBlock],
    image_size: tuple[int, int],
    page_rect: fitz.Rect,
    *,
    padding_px: int = 0,
) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in build_mask_shapes(text_blocks, image_size, page_rect):
        if shape["kind"] == "polygon":
            draw.polygon(shape["points"], fill=255)
        else:
            draw.rectangle(shape["bbox"], fill=255)

    if padding_px <= 0:
        return mask

    mask_array = np.array(mask, dtype=np.uint8)
    kernel_size = max(1, padding_px * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    expanded = cv2.dilate(mask_array, kernel, iterations=1)
    return Image.fromarray(expanded, mode="L")


def render_overlay_background(
    page_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
    *,
    options: ConversionOptions,
) -> BackgroundRenderResult:
    mask_image = build_text_mask_image(
        text_blocks,
        page_image.size,
        page_rect,
        padding_px=max(0, options.inpaint_padding_px),
    )
    mask_array = np.array(mask_image, dtype=np.uint8)
    if np.count_nonzero(mask_array) == 0:
        logger.info("No overlay mask pixels were generated for background rendering")
        return BackgroundRenderResult(
            image=page_image.convert("RGB").copy(),
            engine_name=None,
            note="No overlay mask pixels were generated.",
            mask_image=mask_image,
        )

    engine, note = resolve_background_inpainting_engine(page_image, mask_image, options)
    fallback_engine = (
        OpenCvFastInpaintingEngine()
        if mask_area_ratio(mask_array) <= options.inpaint_max_area_ratio
        else WhiteBoxInpaintingEngine()
    )
    logger.info("Using background engine %s", engine.name)
    logger.debug("Background engine note: %s", note)
    if engine.name == fallback_engine.name:
        return BackgroundRenderResult(
            image=engine.inpaint(page_image, mask_image),
            engine_name=engine.name,
            note=note,
            mask_image=mask_image,
        )

    try:
        rendered_image = engine.inpaint(page_image, mask_image)
        rendered_note = note
    except BackgroundInpaintingError as error:
        logger.warning(
            "Background engine %s failed; falling back to %s: %s",
            engine.name,
            fallback_engine.name,
            error,
        )
        rendered_image = fallback_engine.inpaint(page_image, mask_image)
        rendered_note = f"{note} Fallback to {fallback_engine.name}: {error}"
    return BackgroundRenderResult(
        image=rendered_image,
        engine_name=engine.name if rendered_note == note else fallback_engine.name,
        note=rendered_note,
        mask_image=mask_image,
    )


def mask_area_ratio(mask_array: np.ndarray) -> float:
    return float(np.count_nonzero(mask_array)) / float(mask_array.size)


def estimate_background_complexity(page_image: Image.Image, mask_image: Image.Image) -> float:
    image_array = np.array(page_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    kernel = np.ones((9, 9), dtype=np.uint8)
    outer_ring = cv2.dilate(mask_array, kernel, iterations=1)
    context_ring = np.logical_and(outer_ring > 0, mask_array == 0)
    if not np.any(context_ring):
        context_ring = mask_array == 0
    context_pixels = gray[context_ring]
    if context_pixels.size == 0:
        return 0.0

    luma_std = float(np.std(context_pixels))
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges[context_ring])) / float(context_pixels.size)
    variance_score = min(1.0, luma_std / 64.0)
    edge_score = min(1.0, edge_density / 0.12)
    return round(variance_score * 0.65 + edge_score * 0.35, 4)


def compute_mask_crop_box(mask_image: Image.Image, *, padding_px: int) -> tuple[int, int, int, int] | None:
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    points = cv2.findNonZero(mask_array)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    image_width, image_height = mask_image.size
    left = max(0, x - padding_px)
    top = max(0, y - padding_px)
    right = min(image_width, x + width + padding_px)
    bottom = min(image_height, y + height + padding_px)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def invoke_diffusion_backend(
    crop_image: Image.Image,
    crop_mask: Image.Image,
    *,
    command: str,
    model: str,
    device: str,
    max_crop_edge: int,
    timeout_sec: float,
) -> Image.Image:
    prepared_image = crop_image.convert("RGB")
    prepared_mask = crop_mask.convert("L")
    original_size = prepared_image.size
    longest_edge = max(original_size)
    if longest_edge > max_crop_edge:
        scale = float(max_crop_edge) / float(longest_edge)
        resized_size = (
            max(1, int(round(original_size[0] * scale))),
            max(1, int(round(original_size[1] * scale))),
        )
        prepared_image = prepared_image.resize(resized_size, Image.Resampling.LANCZOS)
        prepared_mask = prepared_mask.resize(resized_size, Image.Resampling.NEAREST)

    with tempfile.TemporaryDirectory(prefix="pdf2ppt-inpaint-") as temp_dir:
        temp_path = Path(temp_dir)
        image_dir = temp_path / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        output_dir = temp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "crop.png"
        mask_path = temp_path / "mask.png"
        prepared_image.save(image_path)
        prepared_mask.save(mask_path)

        command_args = [
            command,
            "run",
            f"--model={model}",
            f"--device={device}",
            f"--image={image_dir}",
            f"--mask={mask_path}",
            f"--output={output_dir}",
        ]
        logger.debug("Invoking diffusion backend: %s", " ".join(command_args))
        try:
            result = subprocess.run(
                command_args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as error:
            raise BackgroundInpaintingError(
                f"Diffusion backend command '{command}' timed out after {timeout_sec:.1f}s."
            ) from error
        except OSError as error:
            raise BackgroundInpaintingError(
                f"Diffusion backend command '{command}' failed to start: {error}"
            ) from error
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip() or "unknown backend failure"
            raise BackgroundInpaintingError(
                f"Diffusion backend command failed with exit code {result.returncode}: {stderr}"
            )

        output_candidates = sorted(output_dir.glob("*.png"))
        if not output_candidates:
            raise BackgroundInpaintingError(
                f"Diffusion backend command '{command}' did not produce any PNG output."
            )

        output_image = Image.open(output_candidates[0]).convert("RGB")
        if output_image.size != original_size:
            output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
        return output_image


def resolve_background_inpainting_engine(
    page_image: Image.Image,
    mask_image: Image.Image,
    options: ConversionOptions,
) -> tuple[BackgroundInpaintingEngine, str]:
    requested_engine = options.inpaint_engine
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    mask_ratio = mask_area_ratio(mask_array)
    diffusion_engine = DiffusionLocalInpaintingEngine(
        command=options.diffusion_command,
        model=options.diffusion_model,
        device=options.diffusion_device,
        max_crop_edge=options.diffusion_max_crop_edge,
        crop_padding_px=max(24, options.inpaint_padding_px * 3),
        timeout_sec=options.diffusion_timeout_sec,
    )
    if requested_engine == "white-box":
        return WhiteBoxInpaintingEngine(), f"Selected white-box engine explicitly (mask area ratio {mask_ratio:.4f})."
    if requested_engine == "opencv-fast":
        return OpenCvFastInpaintingEngine(), (
            f"Selected opencv-fast engine explicitly (mask area ratio {mask_ratio:.4f})."
        )
    if requested_engine == "diffusion-local":
        return diffusion_engine, (
            f"Selected diffusion-local engine explicitly (mask area ratio {mask_ratio:.4f}, "
            f"model {options.diffusion_model}, device {options.diffusion_device})."
        )

    if mask_ratio > options.inpaint_max_area_ratio:
        return WhiteBoxInpaintingEngine(), (
            f"Auto route fell back to white-box because mask area ratio {mask_ratio:.4f} "
            f"exceeded threshold {options.inpaint_max_area_ratio:.4f}."
        )
    complexity = estimate_background_complexity(page_image, mask_image)
    backend_available = shutil.which(options.diffusion_command) is not None
    if backend_available and complexity >= options.diffusion_complexity_threshold:
        return diffusion_engine, (
            f"Auto route selected diffusion-local because complexity score {complexity:.4f} "
            f"met threshold {options.diffusion_complexity_threshold:.4f}."
        )
    if not backend_available and complexity >= options.diffusion_complexity_threshold:
        return OpenCvFastInpaintingEngine(), (
            f"Auto route fell back to opencv-fast because complexity score {complexity:.4f} "
            f"met threshold {options.diffusion_complexity_threshold:.4f} but diffusion command "
            f"'{options.diffusion_command}' was unavailable."
        )
    return OpenCvFastInpaintingEngine(), (
        f"Auto route selected opencv-fast because complexity score {complexity:.4f} "
        f"was below threshold {options.diffusion_complexity_threshold:.4f}."
    )
