from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .inpainting_masks import compute_mask_crop_box

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


__all__ = [
    "BackgroundInpaintingEngine",
    "BackgroundInpaintingError",
    "BackgroundRenderResult",
    "DiffusionLocalInpaintingEngine",
    "OpenCvFastInpaintingEngine",
    "WhiteBoxInpaintingEngine",
    "invoke_diffusion_backend",
]
