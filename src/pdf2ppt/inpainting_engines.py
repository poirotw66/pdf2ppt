from __future__ import annotations

import atexit
import importlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .inpainting_masks import (
    DEFAULT_LOW_TEXTURE_CONTEXT_DILATE_PX,
    DEFAULT_LOW_TEXTURE_EDGE_THRESHOLD,
    DEFAULT_LOW_TEXTURE_STD_THRESHOLD,
    estimate_background_complexity,
    estimate_low_texture_mask_fraction,
    mask_area_ratio,
)

logger = logging.getLogger(__name__)


DEFAULT_SMOOTH_GRADIENT_CONTEXT_GAP_PX = 6
DEFAULT_SMOOTH_GRADIENT_EDGE_THRESHOLD = 0.015
DEFAULT_SMOOTH_GRADIENT_QUADRATIC_RESIDUAL_THRESHOLD = 16.0
DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_MAX_DELTA = 0.0
DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_RESIDUAL_SCALE = 4.0
DEFAULT_RELAXED_QUADRATIC_COLOR_BIAS_MAX_DELTA = 36.0
DEFAULT_RELAXED_QUADRATIC_EDGE_THRESHOLD = 0.03
DEFAULT_RELAXED_QUADRATIC_MIN_WIDTH_PX = 120
DEFAULT_RELAXED_QUADRATIC_MAX_HEIGHT_PX = 72
DEFAULT_RELAXED_QUADRATIC_MIN_ASPECT_RATIO = 3.0
DEFAULT_TELEA_RADIUS_MIN_SCALE = 0.6
DEFAULT_TELEA_RADIUS_MAX_SCALE = 2.5
DEFAULT_TELEA_RADIUS_REFERENCE_SPAN_PX = 48.0
DEFAULT_TELEA_RADIUS_EDGE_DENSITY_THRESHOLD = 0.08
DEFAULT_TELEA_RADIUS_EDGE_DENSITY_MIN_FACTOR = 0.7
DEFAULT_TELEA_SMALL_COMPONENT_MAX_SPAN_PX = 36
DEFAULT_TELEA_GROUP_PROXIMITY_PX = 12
DEFAULT_TELEA_GROUP_PROXIMITY_MIN_SCALE = 0.75
DEFAULT_TELEA_GROUP_PROXIMITY_MAX_SCALE = 2.0
DEFAULT_TELEA_COMPACT_WIDE_MAX_WIDTH_PX = 220
DEFAULT_TELEA_COMPACT_WIDE_MAX_HEIGHT_PX = 64
DEFAULT_TELEA_COMPACT_WIDE_MIN_ASPECT_RATIO = 2.0
DEFAULT_TELEA_COMPACT_WIDE_EDGE_DENSITY_THRESHOLD = 0.05
DEFAULT_TELEA_ISOLATED_LABEL_MAX_WIDTH_PX = 260
DEFAULT_TELEA_ISOLATED_LABEL_MAX_HEIGHT_PX = 80
DEFAULT_TELEA_ISOLATED_LABEL_MIN_ASPECT_RATIO = 2.5
DEFAULT_TELEA_ISOLATED_LABEL_EDGE_DENSITY_THRESHOLD = 0.02
DEFAULT_TELEA_ISOLATED_LABEL_RADIUS_CAP = 5.0
DEFAULT_TELEA_DIRECTIONAL_SIDE_LUMA_MARGIN = 20.0
DEFAULT_TELEA_DIRECTIONAL_SIDE_EDGE_MARGIN = 0.03
DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_LUMA_MARGIN = 40.0
DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_EDGE_MARGIN = 0.05
DEFAULT_STRUCTURAL_LINE_PADDING_PX = 4
DEFAULT_STRUCTURAL_LINE_MIN_KERNEL_PX = 12
DEFAULT_LAMA_ONNX_MAX_SIDE_PX = 1536
DEFAULT_LAMA_MASK_CLOSE_PX = 2
DEFAULT_LAMA_MASK_EXTRA_PADDING_PX = 2
DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX = 4
DEFAULT_LAMA_EDGE_FEATHER_PX = 3
DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD = 0.12
DEFAULT_LAMA_PATCH_MAX_COMPONENTS_FOR_FULL_PAGE = 4
DEFAULT_LAMA_PATCH_CONTEXT_PADDING_PX = 28
DEFAULT_LAMA_PATCH_MERGE_GAP_PX = 18
DEFAULT_LAMA_PATCH_MAX_SIDE_PX = 512
DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD = 0.75
DEFAULT_LAMA_PATCH_HYBRID_COMPLEXITY_THRESHOLD = 0.42
LAMA_INPAINT_ENGINES = frozenset(
    {
        "lama-pytorch",
        "lama-onnx-cuda",
        "lama-pytorch-hybrid",
        "lama-onnx-cuda-hybrid",
    }
)
LAMA_HYBRID_INPAINT_ENGINES = frozenset(
    {
        "lama-pytorch-hybrid",
        "lama-onnx-cuda-hybrid",
    }
)
DEFAULT_LAMA_COMPOSITE_MASK_DILATE_PX = 2
DEFAULT_LAMA_COMPOSITE_HARD_CORE_ERODE_PX = 1
DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA = 1.2
DEFAULT_LAMA_COMPOSITE_ALPHA_GAIN = 2.0
DEFAULT_LAMA_ONNX_MODEL_FILENAMES = (
    "lama_fp16.onnx",
    "big-lama-fp16.onnx",
    "big-lama.onnx",
    "lama.onnx",
    "model.onnx",
)

_LAMA_SESSION_CACHE: dict[tuple[str, str, str], Any] = {}
_LAMA_SESSION_CACHE_LOCK = Lock()


def base_lama_inpaint_engine(engine: str) -> str:
    if engine == "lama-pytorch-hybrid":
        return "lama-pytorch"
    if engine == "lama-onnx-cuda-hybrid":
        return "lama-onnx-cuda"
    return engine


def uses_lama_inpaint_engine(engine: str) -> bool:
    return engine in LAMA_INPAINT_ENGINES


def uses_lama_patch_hybrid(engine: str, *, enabled: bool = True) -> bool:
    return enabled and engine in LAMA_HYBRID_INPAINT_ENGINES


@dataclass(slots=True)
class BackgroundRenderResult:
    image: Image.Image
    engine_name: str | None
    note: str | None
    mask_image: Image.Image | None = None
    debug_images: dict[str, Image.Image] = field(default_factory=dict, repr=False)


class BackgroundInpaintingError(RuntimeError):
    pass


class BackgroundInpaintingEngine:
    name = "base"

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        raise NotImplementedError

    @property
    def last_debug_note(self) -> str | None:
        return None

    @property
    def last_debug_images(self) -> dict[str, Image.Image]:
        return {}


class WhiteBoxInpaintingEngine(BackgroundInpaintingEngine):
    name = "white-box"

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        mask_array = np.array(mask_image.convert("L")) > 0
        result = np.array(page_image.convert("RGB"), copy=True)
        result[mask_array] = 255
        return Image.fromarray(result, mode="RGB")


class OpenCvFastInpaintingEngine(BackgroundInpaintingEngine):
    name = "opencv-fast"

    def __init__(
        self,
        *,
        radius: float = 3.0,
        flat_background_std_threshold: float = DEFAULT_LOW_TEXTURE_STD_THRESHOLD,
        flat_background_edge_threshold: float = DEFAULT_LOW_TEXTURE_EDGE_THRESHOLD,
        context_dilate_px: int = DEFAULT_LOW_TEXTURE_CONTEXT_DILATE_PX,
        context_gap_px: int = DEFAULT_SMOOTH_GRADIENT_CONTEXT_GAP_PX,
        component_expand_px: int = 2,
        smooth_gradient_edge_threshold: float = DEFAULT_SMOOTH_GRADIENT_EDGE_THRESHOLD,
        smooth_gradient_residual_threshold: float = DEFAULT_SMOOTH_GRADIENT_QUADRATIC_RESIDUAL_THRESHOLD,
        smooth_gradient_color_bias_max_delta: float = DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_MAX_DELTA,
        smooth_gradient_color_bias_residual_scale: float = DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_RESIDUAL_SCALE,
        blend_sigma: float = 1.4,
        telea_radius_min_scale: float = DEFAULT_TELEA_RADIUS_MIN_SCALE,
        telea_radius_max_scale: float = DEFAULT_TELEA_RADIUS_MAX_SCALE,
        telea_radius_reference_span_px: float = DEFAULT_TELEA_RADIUS_REFERENCE_SPAN_PX,
        telea_radius_edge_density_threshold: float = DEFAULT_TELEA_RADIUS_EDGE_DENSITY_THRESHOLD,
        telea_radius_edge_density_min_factor: float = DEFAULT_TELEA_RADIUS_EDGE_DENSITY_MIN_FACTOR,
        telea_small_component_max_span_px: int = DEFAULT_TELEA_SMALL_COMPONENT_MAX_SPAN_PX,
        telea_group_proximity_px: int = DEFAULT_TELEA_GROUP_PROXIMITY_PX,
        telea_group_proximity_min_scale: float = DEFAULT_TELEA_GROUP_PROXIMITY_MIN_SCALE,
        telea_group_proximity_max_scale: float = DEFAULT_TELEA_GROUP_PROXIMITY_MAX_SCALE,
    ) -> None:
        self.radius = radius
        self.flat_background_std_threshold = flat_background_std_threshold
        self.flat_background_edge_threshold = flat_background_edge_threshold
        self.context_dilate_px = max(1, context_dilate_px)
        self.context_gap_px = max(0, context_gap_px)
        self.component_expand_px = max(0, component_expand_px)
        self.smooth_gradient_edge_threshold = max(0.0, smooth_gradient_edge_threshold)
        self.smooth_gradient_residual_threshold = max(0.0, smooth_gradient_residual_threshold)
        self.smooth_gradient_color_bias_max_delta = max(0.0, smooth_gradient_color_bias_max_delta)
        self.smooth_gradient_color_bias_residual_scale = max(1e-3, smooth_gradient_color_bias_residual_scale)
        self.blend_sigma = max(0.1, blend_sigma)
        self.telea_radius_min_scale = max(0.1, telea_radius_min_scale)
        self.telea_radius_max_scale = max(self.telea_radius_min_scale, telea_radius_max_scale)
        self.telea_radius_reference_span_px = max(1.0, telea_radius_reference_span_px)
        self.telea_radius_edge_density_threshold = max(1e-3, telea_radius_edge_density_threshold)
        self.telea_radius_edge_density_min_factor = float(np.clip(telea_radius_edge_density_min_factor, 0.1, 1.0))
        self.telea_small_component_max_span_px = max(1, telea_small_component_max_span_px)
        self.telea_group_proximity_px = max(0, telea_group_proximity_px)
        self.telea_group_proximity_min_scale = max(0.1, telea_group_proximity_min_scale)
        self.telea_group_proximity_max_scale = max(self.telea_group_proximity_min_scale, telea_group_proximity_max_scale)
        self._last_debug_note: str | None = None
        self._protected_line_mask: np.ndarray | None = None
        self._last_debug_images: dict[str, Image.Image] = {}

    @property
    def last_debug_note(self) -> str | None:
        return self._last_debug_note

    @property
    def last_debug_images(self) -> dict[str, Image.Image]:
        return self._last_debug_images

    def set_protected_line_mask(self, protected_line_mask: np.ndarray | None) -> None:
        if protected_line_mask is None:
            self._protected_line_mask = None
            return
        self._protected_line_mask = protected_line_mask.astype(bool, copy=True)

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        self._last_debug_note = None
        self._last_debug_images = {}
        mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
        if np.count_nonzero(mask_array) == 0:
            return page_image.convert("RGB").copy()
        source = cv2.cvtColor(np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        protected_prefill_ring_mask = np.zeros(mask_array.shape, dtype=np.uint8)
        prefill_result = _prefill_low_texture_regions(
            source,
            mask_array,
            protected_line_mask=self._protected_line_mask,
            protected_prefill_ring_mask=protected_prefill_ring_mask,
            flat_background_std_threshold=self.flat_background_std_threshold,
            flat_background_edge_threshold=self.flat_background_edge_threshold,
            context_dilate_px=self.context_dilate_px,
            context_gap_px=self.context_gap_px,
            component_expand_px=self.component_expand_px,
            smooth_gradient_edge_threshold=self.smooth_gradient_edge_threshold,
            smooth_gradient_residual_threshold=self.smooth_gradient_residual_threshold,
            smooth_gradient_color_bias_max_delta=self.smooth_gradient_color_bias_max_delta,
            smooth_gradient_color_bias_residual_scale=self.smooth_gradient_color_bias_residual_scale,
        )
        if len(prefill_result) == 2:
            prefilled_source, residual_mask = prefill_result
            prefilled_patches = []
        else:
            prefilled_source, residual_mask, prefilled_patches = prefill_result
        if np.count_nonzero(residual_mask) == 0:
            repaired = prefilled_source
            self._last_debug_note = "Residual Telea groups: 0 after surface prefill."
        else:
            repaired, diagnostics = _inpaint_residual_components(
                prefilled_source,
                residual_mask,
                protected_line_mask=self._protected_line_mask,
                base_radius=self.radius,
                min_radius=self.radius * self.telea_radius_min_scale,
                max_radius=self.radius * self.telea_radius_max_scale,
                reference_span_px=self.telea_radius_reference_span_px,
                edge_density_threshold=self.telea_radius_edge_density_threshold,
                edge_density_min_factor=self.telea_radius_edge_density_min_factor,
                small_component_max_span_px=self.telea_small_component_max_span_px,
                group_proximity_px=self.telea_group_proximity_px,
                group_proximity_min_scale=self.telea_group_proximity_min_scale,
                group_proximity_max_scale=self.telea_group_proximity_max_scale,
            )
            self._last_debug_note = _format_telea_group_diagnostics(diagnostics)
        repaired = _restore_low_texture_regions(
            source,
            repaired,
            mask_array,
            prefilled_patches=prefilled_patches,
            protected_line_mask=self._protected_line_mask,
            flat_background_std_threshold=self.flat_background_std_threshold,
            flat_background_edge_threshold=self.flat_background_edge_threshold,
            context_dilate_px=self.context_dilate_px,
            context_gap_px=self.context_gap_px,
            component_expand_px=self.component_expand_px,
            smooth_gradient_edge_threshold=self.smooth_gradient_edge_threshold,
            smooth_gradient_residual_threshold=self.smooth_gradient_residual_threshold,
            smooth_gradient_color_bias_max_delta=self.smooth_gradient_color_bias_max_delta,
            smooth_gradient_color_bias_residual_scale=self.smooth_gradient_color_bias_residual_scale,
            blend_sigma=self.blend_sigma,
        )
        repaired, restored_line_pixels = _restore_structural_line_regions(source, repaired, mask_array)
        if restored_line_pixels > 0:
            line_note = f"Structural line restore pixels: {restored_line_pixels}."
            self._last_debug_note = f"{self._last_debug_note} {line_note}" if self._last_debug_note else line_note
        if np.any(protected_prefill_ring_mask):
            self._last_debug_images["protected_prefill_ring_mask"] = Image.fromarray(protected_prefill_ring_mask, mode="L")
        return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))


class LamaOnnxCudaInpaintingEngine(BackgroundInpaintingEngine):
    name = "lama-onnx-cuda"

    def __init__(
        self,
        *,
        model_root: Path | None,
        cuda_provider: str = "CUDAExecutionProvider",
        execution_mode: str = "sequential",
        max_side_px: int = DEFAULT_LAMA_ONNX_MAX_SIDE_PX,
        patch_hybrid: bool = True,
    ) -> None:
        self.model_root = model_root
        self.cuda_provider = cuda_provider
        self.execution_mode = execution_mode.lower().strip()
        self.max_side_px = max(256, int(max_side_px))
        self.patch_hybrid = patch_hybrid
        self._last_debug_note: str | None = None

    @property
    def last_debug_note(self) -> str | None:
        return self._last_debug_note

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        self._last_debug_note = None
        source_rgb = np.array(page_image.convert("RGB"), dtype=np.uint8)
        mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
        if np.count_nonzero(mask_array) == 0:
            return Image.fromarray(source_rgb, mode="RGB")

        model_path = _resolve_lama_model_path(self.model_root)
        session = _get_lama_session(
            model_path=model_path,
            cuda_provider=self.cuda_provider,
            execution_mode=self.execution_mode,
        )
        working_mask = _prepare_lama_working_mask(mask_array)
        if _should_use_lama_patch_inpaint(working_mask):
            result, patch_note = _inpaint_lama_page_by_patches(
                source_rgb,
                working_mask,
                lambda crop_source, crop_mask: self._inpaint_onnx_crop(
                    session,
                    crop_source,
                    crop_mask,
                    max_side_px=DEFAULT_LAMA_PATCH_MAX_SIDE_PX,
                )[0],
                use_patch_hybrid=self.patch_hybrid,
            )
            composite_note = _lama_composite_debug_suffix(dilate_px=DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX)
            self._last_debug_note = (
                f"LaMa ONNX CUDA provider={self.cuda_provider} execution_mode={self.execution_mode} "
                f"model={model_path.name} {patch_note}{composite_note}."
            )
            return Image.fromarray(result, mode="RGB")

        resize_note = ""
        fixed_input_note = ""
        if self.patch_hybrid and _should_use_opencv_for_lama_patch(source_rgb, working_mask):
            restored_rgb = _inpaint_opencv_fast_crop(source_rgb, working_mask)
            result = _finalize_lama_inpaint(source_rgb, restored_rgb, working_mask)
            hybrid_note = " full-page-hybrid-opencv"
        else:
            restored_rgb, resize_note, fixed_input_note = self._inpaint_onnx_crop(session, source_rgb, working_mask)
            result = _finalize_lama_inpaint(source_rgb, restored_rgb, working_mask)
            hybrid_note = ""
        composite_note = _lama_composite_debug_suffix(dilate_px=DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX)
        self._last_debug_note = (
            f"LaMa ONNX CUDA provider={self.cuda_provider} execution_mode={self.execution_mode} "
            f"model={model_path.name}{resize_note}{fixed_input_note}{hybrid_note}{composite_note}."
        )
        return Image.fromarray(result, mode="RGB")

    def _inpaint_onnx_crop(
        self,
        session: Any,
        source_rgb: np.ndarray,
        working_mask: np.ndarray,
        *,
        max_side_px: int | None = None,
    ) -> tuple[np.ndarray, str, str]:
        inference_mask = _expand_lama_inference_mask(
            working_mask,
            dilate_px=DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX,
        )
        resized_rgb, resized_mask, resized = _resize_lama_inputs(
            source_rgb,
            inference_mask,
            max_side_px=max_side_px if max_side_px is not None else self.max_side_px,
        )
        fixed_input_size = _resolve_lama_fixed_input_size(session)
        model_rgb, model_mask, fit_to_fixed_size = _fit_lama_inputs_to_model(
            resized_rgb,
            resized_mask,
            fixed_input_size,
        )
        if fixed_input_size is None:
            model_rgb, model_mask = _pad_lama_inputs(model_rgb, model_mask)
        image_tensor = np.transpose(model_rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        mask_tensor = (model_mask > 0).astype(np.float32)[None, None, ...]

        model_inputs = _build_lama_model_inputs(session, image_tensor, mask_tensor)
        try:
            outputs = session.run(None, model_inputs)
        except Exception as error:
            raise BackgroundInpaintingError(f"LaMa ONNX inference failed: {error}") from error
        if not outputs:
            raise BackgroundInpaintingError("LaMa ONNX inference returned no outputs.")

        restored_rgb = _normalize_lama_output(outputs[0])
        restored_rgb = restored_rgb[: model_rgb.shape[0], : model_rgb.shape[1], :]
        if fit_to_fixed_size:
            restored_rgb = cv2.resize(
                restored_rgb,
                (resized_rgb.shape[1], resized_rgb.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            restored_rgb = restored_rgb[: resized_rgb.shape[0], : resized_rgb.shape[1], :]
        if resized:
            restored_rgb = cv2.resize(
                restored_rgb,
                (source_rgb.shape[1], source_rgb.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        resize_note = f" resized-to-max-side={self.max_side_px}" if resized else ""
        fixed_input_note = (
            f" model_input={fixed_input_size[1]}x{fixed_input_size[0]}"
            if fixed_input_size is not None
            else ""
        )
        return restored_rgb, resize_note, fixed_input_note


@dataclass(slots=True)
class _LamaPytorchWorkerHandle:
    process: subprocess.Popen[str]
    lock: Lock


_LAMA_PYTORCH_WORKERS: dict[tuple[str, str, str], _LamaPytorchWorkerHandle] = {}
_LAMA_PYTORCH_WORKERS_LOCK = Lock()


def _shutdown_lama_pytorch_workers() -> None:
    with _LAMA_PYTORCH_WORKERS_LOCK:
        handles = list(_LAMA_PYTORCH_WORKERS.values())
        _LAMA_PYTORCH_WORKERS.clear()
    for handle in handles:
        with handle.lock:
            if handle.process.poll() is None:
                try:
                    handle.process.stdin.write("shutdown\n")
                    handle.process.stdin.flush()
                except OSError:
                    pass
                handle.process.terminate()


atexit.register(_shutdown_lama_pytorch_workers)


class LamaPytorchInpaintingEngine(BackgroundInpaintingEngine):
    name = "lama-pytorch"

    def __init__(
        self,
        *,
        model_root: Path | None,
        repo_root: Path | None,
        device: str = "cuda",
        python_executable: Path | None = None,
        max_side_px: int = DEFAULT_LAMA_ONNX_MAX_SIDE_PX,
        patch_hybrid: bool = True,
    ) -> None:
        self.model_root = model_root
        self.repo_root = repo_root
        self.device = device.strip() or "cuda"
        self.python_executable = python_executable
        self.max_side_px = max(256, int(max_side_px))
        self.patch_hybrid = patch_hybrid
        self._last_debug_note: str | None = None

    @property
    def last_debug_note(self) -> str | None:
        return self._last_debug_note

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        self._last_debug_note = None
        source_rgb = np.array(page_image.convert("RGB"), dtype=np.uint8)
        mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
        if np.count_nonzero(mask_array) == 0:
            return Image.fromarray(source_rgb, mode="RGB")

        model_path = _resolve_lama_pytorch_model_path(self.model_root)
        repo_root = _resolve_lama_repo_root(self.repo_root)
        python_executable = _resolve_lama_python_executable(self.python_executable)
        _validate_lama_pytorch_runtime(python_executable=python_executable, repo_root=repo_root)

        working_mask = _prepare_lama_working_mask(mask_array)
        if _should_use_lama_patch_inpaint(working_mask):
            result, patch_note = _inpaint_lama_page_by_patches(
                source_rgb,
                working_mask,
                lambda crop_source, crop_mask: self._inpaint_pytorch_crop(
                    crop_source,
                    crop_mask,
                    model_path=model_path,
                    repo_root=repo_root,
                    python_executable=python_executable,
                    max_side_px=DEFAULT_LAMA_PATCH_MAX_SIDE_PX,
                )[0],
                use_patch_hybrid=self.patch_hybrid,
            )
            composite_note = _lama_composite_debug_suffix(dilate_px=DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX)
            self._last_debug_note = (
                f"Official LaMa repo={repo_root.name} device={self.device} model={model_path.name} "
                f"python={Path(python_executable).name} persistent-worker {patch_note}{composite_note}."
            )
            return Image.fromarray(result, mode="RGB")

        resize_note = ""
        if self.patch_hybrid and _should_use_opencv_for_lama_patch(source_rgb, working_mask):
            restored_rgb = _inpaint_opencv_fast_crop(source_rgb, working_mask)
            result = _finalize_lama_inpaint(source_rgb, restored_rgb, working_mask)
            hybrid_note = " full-page-hybrid-opencv"
        else:
            restored_rgb, resize_note = self._inpaint_pytorch_crop(
                source_rgb,
                working_mask,
                model_path=model_path,
                repo_root=repo_root,
                python_executable=python_executable,
            )
            result = _finalize_lama_inpaint(source_rgb, restored_rgb, working_mask)
            hybrid_note = ""
        composite_note = _lama_composite_debug_suffix(dilate_px=DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX)
        self._last_debug_note = (
            f"Official LaMa repo={repo_root.name} device={self.device} model={model_path.name} "
            f"python={Path(python_executable).name} persistent-worker{resize_note}{hybrid_note}{composite_note}."
        )
        return Image.fromarray(result, mode="RGB")

    def _inpaint_pytorch_crop(
        self,
        source_rgb: np.ndarray,
        working_mask: np.ndarray,
        *,
        model_path: Path,
        repo_root: Path,
        python_executable: Path,
        max_side_px: int | None = None,
    ) -> tuple[np.ndarray, str]:
        inference_mask = _expand_lama_inference_mask(
            working_mask,
            dilate_px=DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX,
        )
        resized_rgb, resized_inference_mask, resized = _resize_lama_inputs(
            source_rgb,
            inference_mask,
            max_side_px=max_side_px if max_side_px is not None else self.max_side_px,
        )
        resized_page = Image.fromarray(resized_rgb, mode="RGB")
        resized_mask_image = Image.fromarray((resized_inference_mask > 0).astype(np.uint8) * 255, mode="L")

        with tempfile.TemporaryDirectory(prefix="pdf2ppt-lama-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            indir = temp_dir / "inputs"
            outdir = temp_dir / "outputs"
            indir.mkdir(parents=True, exist_ok=True)
            outdir.mkdir(parents=True, exist_ok=True)

            image_path = indir / "page.png"
            mask_path = indir / "page_mask001.png"
            resized_page.save(image_path)
            resized_mask_image.save(mask_path)

            _run_lama_pytorch_prediction(
                python_executable=python_executable,
                repo_root=repo_root,
                model_path=model_path,
                device=self.device,
                indir=indir,
                outdir=outdir,
            )

            output_path = outdir / "page_mask001.png"
            if not output_path.exists():
                output_candidates = sorted(outdir.rglob("*.png"))
                if len(output_candidates) == 1:
                    output_path = output_candidates[0]
                else:
                    raise BackgroundInpaintingError(
                        f"Official LaMa prediction did not produce the expected output under {outdir}."
                    )

            restored_rgb = np.array(Image.open(output_path).convert("RGB"), dtype=np.uint8)
            if restored_rgb.shape[:2] != resized_rgb.shape[:2]:
                raise BackgroundInpaintingError(
                    f"Official LaMa output shape {restored_rgb.shape[:2]} did not match input shape {resized_rgb.shape[:2]}."
                )
            if resized:
                restored_rgb = cv2.resize(
                    restored_rgb,
                    (source_rgb.shape[1], source_rgb.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
        resize_note = f" resized-to-max-side={self.max_side_px}" if resized else ""
        return restored_rgb, resize_note


def _resolve_lama_model_path(model_root: Path | None) -> Path:
    if model_root is None:
        raise BackgroundInpaintingError("LaMa ONNX model path is not configured. Set inpaint_model_root.")
    resolved_root = model_root.expanduser().resolve()
    if resolved_root.is_file():
        if resolved_root.suffix.lower() != ".onnx":
            raise BackgroundInpaintingError(f"LaMa model file must be an .onnx file: {resolved_root}")
        return resolved_root
    if not resolved_root.exists():
        raise BackgroundInpaintingError(f"LaMa model root does not exist: {resolved_root}")
    for candidate_name in DEFAULT_LAMA_ONNX_MODEL_FILENAMES:
        candidate_path = resolved_root / candidate_name
        if candidate_path.exists():
            return candidate_path
    onnx_files = sorted(resolved_root.glob("*.onnx"))
    if len(onnx_files) == 1:
        return onnx_files[0]
    raise BackgroundInpaintingError(
        f"No LaMa ONNX model was found under {resolved_root}. Expected one of: "
        f"{', '.join(DEFAULT_LAMA_ONNX_MODEL_FILENAMES)}"
    )


def _resolve_lama_repo_root(repo_root: Path | None) -> Path:
    if repo_root is None:
        raise BackgroundInpaintingError("Official LaMa repo path is not configured. Set inpaint_lama_repo_root.")
    resolved_root = repo_root.expanduser().resolve()
    predict_script = resolved_root / "bin" / "predict.py"
    if not predict_script.exists():
        raise BackgroundInpaintingError(
            f"Official LaMa repo root is invalid: {resolved_root}. Expected bin/predict.py to exist."
        )
    return resolved_root


def _resolve_lama_pytorch_model_path(model_root: Path | None) -> Path:
    if model_root is None:
        raise BackgroundInpaintingError("Official LaMa model path is not configured. Set inpaint_model_root.")
    resolved_root = model_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise BackgroundInpaintingError(
            f"Official LaMa model path must point to the extracted checkpoint directory: {resolved_root}"
        )
    if not (resolved_root / "config.yaml").exists() or not (resolved_root / "models").is_dir():
        raise BackgroundInpaintingError(
            f"Official LaMa model directory is missing config.yaml or models/: {resolved_root}"
        )
    return resolved_root


def _resolve_lama_python_executable(python_executable: Path | None) -> Path:
    if python_executable is not None:
        resolved_python = python_executable.expanduser().resolve()
        if not resolved_python.exists():
            raise BackgroundInpaintingError(f"Official LaMa python executable does not exist: {resolved_python}")
        return resolved_python

    env_override = os.environ.get("PDF2PPT_LAMA_PYTHON")
    if env_override:
        resolved_python = Path(env_override).expanduser().resolve()
        if resolved_python.exists():
            return resolved_python

    for candidate in _default_lama_python_candidates():
        if candidate.exists():
            return candidate

    return Path(sys.executable)


_LAMA_PYTORCH_VALIDATED: set[tuple[str, str]] = set()
_LAMA_PYTORCH_VALIDATE_LOCK = Lock()


def _default_lama_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).name == "lama":
        candidates.append(Path(conda_prefix) / "bin" / "python")
    home = Path.home()
    for conda_root in (home / "miniconda3", home / "anaconda3", home / "mambaforge", home / "miniforge3"):
        candidates.append(conda_root / "envs" / "lama" / "bin" / "python")
    return candidates


def _validate_lama_pytorch_runtime(*, python_executable: Path, repo_root: Path) -> None:
    cache_key = (str(python_executable), str(repo_root))
    with _LAMA_PYTORCH_VALIDATE_LOCK:
        if cache_key in _LAMA_PYTORCH_VALIDATED:
            return

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    probe = (
        "import saicinpainting.evaluation.utils; "
        "import saicinpainting.training.data.datasets; "
        "import torch"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", probe],
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "import probe failed").strip()
        lines = detail.splitlines()
        concise = "\n".join(lines[-8:]) if len(lines) > 8 else detail
        raise BackgroundInpaintingError(
            f"LaMa PyTorch runtime is not ready in {python_executable}. "
            "The official advimman/lama repo needs its own Python environment "
            "(see lama/requirements.txt or: conda env create -f lama/conda_env.yml). "
            "Pass --inpaint-lama-python or set PDF2PPT_LAMA_PYTHON to a compatible interpreter. "
            f"Import error: {concise}"
        )

    with _LAMA_PYTORCH_VALIDATE_LOCK:
        _LAMA_PYTORCH_VALIDATED.add(cache_key)


def _build_lama_subprocess_env(*, repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["TORCH_HOME"] = str(repo_root)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    return env


def _read_lama_pytorch_response(process: subprocess.Popen[str]) -> dict[str, object]:
    if process.stdout is None:
        raise BackgroundInpaintingError("LaMa PyTorch worker did not expose stdout.")
    while True:
        response_line = process.stdout.readline()
        if not response_line:
            stderr_tail = ""
            if process.stderr is not None:
                stderr_tail = process.stderr.read()[-2000:]
            raise BackgroundInpaintingError(
                f"LaMa PyTorch worker exited unexpectedly while waiting for a response. {stderr_tail}".strip()
            )
        stripped = response_line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
            break
        except json.JSONDecodeError as error:
            raise BackgroundInpaintingError(f"LaMa PyTorch worker returned invalid JSON: {response_line!r}") from error
    if not isinstance(payload, dict):
        raise BackgroundInpaintingError(f"LaMa PyTorch worker returned an unexpected payload: {payload!r}")
    return payload


def _get_lama_pytorch_worker(
    *,
    python_executable: Path,
    repo_root: Path,
    model_path: Path,
    device: str,
) -> _LamaPytorchWorkerHandle:
    cache_key = (str(python_executable), str(model_path), device)
    env = _build_lama_subprocess_env(repo_root=repo_root)
    with _LAMA_PYTORCH_WORKERS_LOCK:
        cached = _LAMA_PYTORCH_WORKERS.get(cache_key)
        if cached is not None and cached.process.poll() is None:
            return cached
        if cached is not None:
            cached.process.kill()

        process = subprocess.Popen(
            [
                str(python_executable),
                "bin/pdf2ppt_predict_server.py",
                f"--model-path={model_path}",
                f"--device={device}",
            ],
            cwd=repo_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        handle = _LamaPytorchWorkerHandle(process=process, lock=Lock())
        _LAMA_PYTORCH_WORKERS[cache_key] = handle

    with handle.lock:
        payload = _read_lama_pytorch_response(handle.process)
        if not payload.get("ok"):
            message = str(payload.get("message", "failed to start LaMa PyTorch worker"))
            raise BackgroundInpaintingError(f"LaMa PyTorch worker failed to start: {message}")
    return handle


def _run_lama_pytorch_prediction(
    *,
    python_executable: Path,
    repo_root: Path,
    model_path: Path,
    device: str,
    indir: Path,
    outdir: Path,
) -> None:
    try:
        worker = _get_lama_pytorch_worker(
            python_executable=python_executable,
            repo_root=repo_root,
            model_path=model_path,
            device=device,
        )
        with worker.lock:
            if worker.process.stdin is None:
                raise BackgroundInpaintingError("LaMa PyTorch worker did not expose stdin.")
            job = {"indir": str(indir), "outdir": str(outdir), "img_suffix": ".png"}
            worker.process.stdin.write(json.dumps(job) + "\n")
            worker.process.stdin.flush()
            payload = _read_lama_pytorch_response(worker.process)
        if not payload.get("ok"):
            message = str(payload.get("message", "official LaMa prediction failed"))
            raise BackgroundInpaintingError(f"Official LaMa prediction failed: {message}")
        return
    except BackgroundInpaintingError:
        raise
    except Exception as error:
        logger.warning("LaMa PyTorch persistent worker failed; falling back to one-shot predict.py: %s", error)

    command = [
        str(python_executable),
        "bin/predict.py",
        f"model.path={model_path}",
        f"indir={indir}",
        f"outdir={outdir}",
        "dataset.img_suffix=.png",
        f"device={device}",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=_build_lama_subprocess_env(repo_root=repo_root),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "official LaMa prediction failed").strip()
        raise BackgroundInpaintingError(f"Official LaMa prediction failed: {detail}")


def _import_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as error:
        raise BackgroundInpaintingError(
            "LaMa ONNX CUDA requires onnxruntime-gpu. Install pdf2ppt[gpu] to enable this engine."
        ) from error


def _get_lama_session(*, model_path: Path, cuda_provider: str, execution_mode: str) -> Any:
    cache_key = (str(model_path), cuda_provider, execution_mode)
    with _LAMA_SESSION_CACHE_LOCK:
        cached_session = _LAMA_SESSION_CACHE.get(cache_key)
    if cached_session is not None:
        return cached_session

    ort = _import_onnxruntime()
    available_providers = set(ort.get_available_providers())
    if cuda_provider not in available_providers:
        raise BackgroundInpaintingError(
            f"LaMa ONNX CUDA provider {cuda_provider!r} is unavailable. "
            f"Available providers: {sorted(available_providers)}"
        )

    session_options = ort.SessionOptions()
    if execution_mode == "parallel":
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[cuda_provider],
        )
    except Exception as error:
        raise BackgroundInpaintingError(f"Failed to initialize LaMa ONNX session: {error}") from error

    with _LAMA_SESSION_CACHE_LOCK:
        _LAMA_SESSION_CACHE.setdefault(cache_key, session)
        return _LAMA_SESSION_CACHE[cache_key]


def _prepare_lama_working_mask(mask_array: np.ndarray, *, close_px: int = DEFAULT_LAMA_MASK_CLOSE_PX) -> np.ndarray:
    binary_mask = (mask_array > 0).astype(np.uint8)
    if not np.any(binary_mask):
        return mask_array
    if close_px > 0:
        kernel_size = close_px * 2 + 1
        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
    return binary_mask * 255


def _should_use_lama_patch_inpaint(mask_array: np.ndarray) -> bool:
    if mask_area_ratio(mask_array) > DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD:
        return True
    component_count, _, _, _ = cv2.connectedComponentsWithStats((mask_array > 0).astype(np.uint8), 8)
    return component_count > DEFAULT_LAMA_PATCH_MAX_COMPONENTS_FOR_FULL_PAGE + 1


def _boxes_overlap_with_gap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    *,
    gap_px: int,
) -> bool:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    return not (
        left_x1 + gap_px < right_x0
        or right_x1 + gap_px < left_x0
        or left_y1 + gap_px < right_y0
        or right_y1 + gap_px < left_y0
    )


def _merge_lama_patch_boxes(
    boxes: list[tuple[int, int, int, int, int]],
    *,
    merge_gap_px: int,
) -> list[tuple[int, int, int, int, list[int]]]:
    if not boxes:
        return []

    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left_box in enumerate(boxes):
        for right_index in range(left_index + 1, len(boxes)):
            if _boxes_overlap_with_gap(left_box[:4], boxes[right_index][:4], gap_px=merge_gap_px):
                union(left_index, right_index)

    grouped: dict[int, list[int]] = {}
    for index, box in enumerate(boxes):
        grouped.setdefault(find(index), []).append(box[4])

    merged_boxes: list[tuple[int, int, int, int, list[int]]] = []
    for member_indices in grouped.values():
        member_boxes = [box for box in boxes if box[4] in member_indices]
        x0 = min(box[0] for box in member_boxes)
        y0 = min(box[1] for box in member_boxes)
        x1 = max(box[2] for box in member_boxes)
        y1 = max(box[3] for box in member_boxes)
        merged_boxes.append((x0, y0, x1, y1, member_indices))
    return merged_boxes


def _build_lama_patch_groups(
    working_mask: np.ndarray,
    *,
    context_padding_px: int,
    merge_gap_px: int,
) -> list[tuple[int, int, int, int, np.ndarray]]:
    height, width = working_mask.shape[:2]
    binary_mask = (working_mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    if component_count <= 1:
        return []

    initial_boxes: list[tuple[int, int, int, int, int]] = []
    for component_index in range(1, component_count):
        x, y, box_width, box_height, _ = stats[component_index]
        if box_width <= 0 or box_height <= 0:
            continue
        initial_boxes.append(
            (
                max(0, int(x) - context_padding_px),
                max(0, int(y) - context_padding_px),
                min(width, int(x + box_width) + context_padding_px),
                min(height, int(y + box_height) + context_padding_px),
                component_index,
            )
        )

    merged_boxes = _merge_lama_patch_boxes(initial_boxes, merge_gap_px=merge_gap_px)
    patch_groups: list[tuple[int, int, int, int, np.ndarray]] = []
    for x0, y0, x1, y1, member_indices in merged_boxes:
        group_mask = np.zeros((height, width), dtype=np.uint8)
        for component_index in member_indices:
            group_mask[labels == component_index] = 255
        patch_groups.append((x0, y0, x1, y1, group_mask))
    return patch_groups


def _composite_lama_patch_into_page(
    page_rgb: np.ndarray,
    patch_rgb: np.ndarray,
    crop_mask: np.ndarray,
    *,
    origin: tuple[int, int],
) -> None:
    x0, y0 = origin
    patch_height, patch_width = patch_rgb.shape[:2]
    region = page_rgb[y0 : y0 + patch_height, x0 : x0 + patch_width]
    blended = _composite_lama_restoration(
        region,
        patch_rgb,
        crop_mask,
        blend_sigma=DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
        alpha_gain=1.35,
        composite_dilate_px=1,
        hard_core_erode_px=0,
    )
    page_rgb[y0 : y0 + patch_height, x0 : x0 + patch_width] = blended


def _should_use_opencv_for_lama_patch(
    crop_source: np.ndarray,
    crop_mask: np.ndarray,
) -> bool:
    crop_page = Image.fromarray(crop_source, mode="RGB")
    crop_mask_image = Image.fromarray(crop_mask, mode="L")
    low_texture_fraction = estimate_low_texture_mask_fraction(crop_page, crop_mask_image)
    if low_texture_fraction >= DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD:
        return True
    complexity = estimate_background_complexity(crop_page, crop_mask_image)
    return (
        complexity <= DEFAULT_LAMA_PATCH_HYBRID_COMPLEXITY_THRESHOLD
        and mask_area_ratio(crop_mask) <= 0.25
    )


def _inpaint_opencv_fast_crop(source_rgb: np.ndarray, working_mask: np.ndarray) -> np.ndarray:
    crop_page = Image.fromarray(source_rgb, mode="RGB")
    crop_mask_image = Image.fromarray(working_mask, mode="L")
    return np.array(OpenCvFastInpaintingEngine().inpaint(crop_page, crop_mask_image), dtype=np.uint8)


def _inpaint_lama_crop_with_hybrid(
    crop_source: np.ndarray,
    crop_working_mask: np.ndarray,
    run_lama_crop_inpaint: Any,
    *,
    use_patch_hybrid: bool,
) -> tuple[np.ndarray, str]:
    if use_patch_hybrid and _should_use_opencv_for_lama_patch(crop_source, crop_working_mask):
        return _inpaint_opencv_fast_crop(crop_source, crop_working_mask), "opencv-fast"
    return run_lama_crop_inpaint(crop_source, crop_working_mask), "lama"


def _inpaint_lama_page_by_patches(
    source_rgb: np.ndarray,
    working_mask: np.ndarray,
    run_crop_inpaint: Any,
    *,
    use_patch_hybrid: bool,
) -> tuple[np.ndarray, str]:
    patch_groups = _build_lama_patch_groups(
        working_mask,
        context_padding_px=DEFAULT_LAMA_PATCH_CONTEXT_PADDING_PX,
        merge_gap_px=DEFAULT_LAMA_PATCH_MERGE_GAP_PX,
    )
    if not patch_groups:
        return source_rgb.copy(), "patch-mode groups=0"

    result = source_rgb.copy()
    opencv_patch_count = 0
    lama_patch_count = 0
    for x0, y0, x1, y1, group_mask in patch_groups:
        crop_source = source_rgb[y0:y1, x0:x1]
        crop_mask = group_mask[y0:y1, x0:x1]
        if not np.any(crop_mask):
            continue
        crop_working_mask = _prepare_lama_working_mask(crop_mask)
        crop_restored, patch_engine = _inpaint_lama_crop_with_hybrid(
            crop_source,
            crop_working_mask,
            run_crop_inpaint,
            use_patch_hybrid=use_patch_hybrid,
        )
        if patch_engine == "opencv-fast":
            opencv_patch_count += 1
        else:
            lama_patch_count += 1
        _composite_lama_patch_into_page(
            result,
            crop_restored,
            crop_working_mask,
            origin=(x0, y0),
        )

    hybrid_note = (
        f" hybrid-opencv={opencv_patch_count} hybrid-lama={lama_patch_count}"
        if use_patch_hybrid
        else ""
    )
    patch_note = (
        f"patch-mode groups={len(patch_groups)}{hybrid_note} "
        f"mask-ratio={mask_area_ratio(working_mask):.4f}"
    )
    return result, patch_note


def _finalize_lama_inpaint(
    source_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    working_mask: np.ndarray,
) -> np.ndarray:
    return _composite_lama_restoration(
        source_rgb,
        restored_rgb,
        working_mask,
        blend_sigma=DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
        alpha_gain=1.35,
        composite_dilate_px=1,
        hard_core_erode_px=0,
    )


def _expand_lama_inference_mask(mask_array: np.ndarray, *, dilate_px: int) -> np.ndarray:
    binary_mask = (mask_array > 0).astype(np.uint8)
    if dilate_px <= 0 or not np.any(binary_mask):
        return mask_array
    kernel_size = dilate_px * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(binary_mask, kernel, iterations=1) * 255


def _composite_lama_restoration(
    source_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    composite_mask: np.ndarray,
    *,
    inference_mask: np.ndarray | None = None,
    blend_sigma: float = DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
    alpha_gain: float = DEFAULT_LAMA_COMPOSITE_ALPHA_GAIN,
    composite_dilate_px: int = DEFAULT_LAMA_COMPOSITE_MASK_DILATE_PX,
    hard_core_erode_px: int = DEFAULT_LAMA_COMPOSITE_HARD_CORE_ERODE_PX,
) -> np.ndarray:
    if inference_mask is not None:
        return _composite_lama_restoration_with_inference_mask(
            source_rgb,
            restored_rgb,
            inference_mask,
            edge_feather_px=DEFAULT_LAMA_EDGE_FEATHER_PX,
        )
    alpha = _build_lama_composite_alpha(
        composite_mask,
        blend_sigma=blend_sigma,
        alpha_gain=alpha_gain,
        composite_dilate_px=composite_dilate_px,
        hard_core_erode_px=hard_core_erode_px,
    )
    if alpha is None:
        return source_rgb
    blended = source_rgb.astype(np.float32) * (1.0 - alpha) + restored_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _composite_lama_restoration_with_inference_mask(
    source_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    inference_mask: np.ndarray,
    *,
    edge_feather_px: int,
) -> np.ndarray:
    binary_mask = (inference_mask > 0).astype(np.uint8)
    if not np.any(binary_mask):
        return source_rgb

    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    feather = max(1, edge_feather_px)
    alpha = np.zeros_like(distance, dtype=np.float32)
    if max_distance <= feather:
        alpha[binary_mask > 0] = 1.0
    else:
        inner_threshold = max_distance - feather
        inside = binary_mask > 0
        alpha[inside & (distance >= inner_threshold)] = 1.0
        edge_band = inside & (distance < inner_threshold)
        alpha[edge_band] = np.clip(distance[edge_band] / feather, 0.0, 1.0)

    alpha = alpha[..., None]
    blended = source_rgb.astype(np.float32) * (1.0 - alpha) + restored_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _build_lama_composite_alpha(
    composite_mask: np.ndarray,
    *,
    blend_sigma: float,
    alpha_gain: float,
    composite_dilate_px: int,
    hard_core_erode_px: int,
) -> np.ndarray | None:
    binary_mask = (composite_mask > 0).astype(np.uint8)
    if not np.any(binary_mask):
        return None

    expanded_mask = binary_mask
    if composite_dilate_px > 0:
        kernel_size = composite_dilate_px * 2 + 1
        expanded_mask = cv2.dilate(
            binary_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )

    alpha = cv2.GaussianBlur(
        expanded_mask.astype(np.float32),
        (0, 0),
        sigmaX=blend_sigma,
        sigmaY=blend_sigma,
    )
    alpha = np.clip(alpha * alpha_gain, 0.0, 1.0)

    if hard_core_erode_px > 0:
        core_kernel_size = hard_core_erode_px * 2 + 1
        hard_core = cv2.erode(
            binary_mask,
            np.ones((core_kernel_size, core_kernel_size), dtype=np.uint8),
            iterations=1,
        )
        if np.any(hard_core):
            alpha[hard_core > 0] = 1.0

    return alpha[..., None]


def _lama_composite_debug_suffix(*, dilate_px: int) -> str:
    return (
        f" mask-close={DEFAULT_LAMA_MASK_CLOSE_PX}"
        f" inference-mask-dilate={dilate_px}"
        f" patch-threshold={DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD:.2f}"
        f" patch-hybrid-low-texture>={DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD:.2f}"
        f" patch-max-side={DEFAULT_LAMA_PATCH_MAX_SIDE_PX}"
    )


def _resize_lama_inputs(
    source_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    max_side_px: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    height, width = source_rgb.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side_px:
        return source_rgb, mask_array, False
    scale = max_side_px / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_rgb = cv2.resize(source_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask_array, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    return resized_rgb, resized_mask, True


def _pad_lama_inputs(source_rgb: np.ndarray, mask_array: np.ndarray, *, stride: int = 8) -> tuple[np.ndarray, np.ndarray]:
    height, width = source_rgb.shape[:2]
    padded_height = int(math.ceil(height / stride) * stride)
    padded_width = int(math.ceil(width / stride) * stride)
    if padded_height == height and padded_width == width:
        return source_rgb, mask_array
    padded_rgb = np.pad(
        source_rgb,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="edge",
    )
    padded_mask = np.pad(
        mask_array,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="constant",
    )
    return padded_rgb, padded_mask


def _resolve_lama_fixed_input_size(session: Any) -> tuple[int, int] | None:
    session_inputs = list(session.get_inputs())
    resolved_shapes: dict[str, tuple[int, int]] = {}
    for model_input in session_inputs:
        shape = getattr(model_input, "shape", None)
        if not isinstance(shape, (list, tuple)) or len(shape) < 4:
            continue
        input_height = _coerce_lama_dimension(shape[-2])
        input_width = _coerce_lama_dimension(shape[-1])
        if input_height is None or input_width is None:
            continue
        lowered_name = model_input.name.lower()
        if "image" in lowered_name:
            resolved_shapes["image"] = (input_height, input_width)
        elif "mask" in lowered_name:
            resolved_shapes["mask"] = (input_height, input_width)

    image_shape = resolved_shapes.get("image")
    mask_shape = resolved_shapes.get("mask")
    if image_shape is not None and mask_shape is not None and image_shape != mask_shape:
        raise BackgroundInpaintingError(
            "LaMa ONNX model exposes incompatible fixed image/mask input sizes."
        )
    return image_shape or mask_shape


def _coerce_lama_dimension(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def _fit_lama_inputs_to_model(
    source_rgb: np.ndarray,
    mask_array: np.ndarray,
    fixed_input_size: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if fixed_input_size is None:
        return source_rgb, mask_array, False
    target_height, target_width = fixed_input_size
    height, width = source_rgb.shape[:2]
    if height == target_height and width == target_width:
        return source_rgb, mask_array, False
    interpolation = cv2.INTER_AREA if target_height <= height and target_width <= width else cv2.INTER_LINEAR
    resized_rgb = cv2.resize(source_rgb, (target_width, target_height), interpolation=interpolation)
    resized_mask = cv2.resize(mask_array, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return resized_rgb, resized_mask, True


def _build_lama_model_inputs(session: Any, image_tensor: np.ndarray, mask_tensor: np.ndarray) -> dict[str, np.ndarray]:
    session_inputs = list(session.get_inputs())
    if len(session_inputs) < 2:
        raise BackgroundInpaintingError("LaMa ONNX model must expose image and mask inputs.")
    resolved_inputs: dict[str, np.ndarray] = {}
    remaining_inputs = session_inputs.copy()
    for keyword, tensor in (("mask", mask_tensor), ("image", image_tensor)):
        for model_input in list(remaining_inputs):
            if keyword in model_input.name.lower():
                resolved_inputs[model_input.name] = tensor
                remaining_inputs.remove(model_input)
                break
    assigned_tensor_ids = {id(tensor) for tensor in resolved_inputs.values()}
    unresolved_tensors = [tensor for tensor in (image_tensor, mask_tensor) if id(tensor) not in assigned_tensor_ids]
    for model_input, tensor in zip(remaining_inputs, unresolved_tensors):
        resolved_inputs[model_input.name] = tensor
    if len(resolved_inputs) < 2:
        raise BackgroundInpaintingError("LaMa ONNX model input mapping failed.")
    return resolved_inputs


def _normalize_lama_output(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[2] not in (1, 3):
        raise BackgroundInpaintingError(f"Unexpected LaMa ONNX output shape: {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.5:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
    else:
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def _prefill_low_texture_regions(
    source: np.ndarray,
    mask_array: np.ndarray,
    *,
    protected_line_mask: np.ndarray | None,
    protected_prefill_ring_mask: np.ndarray | None,
    flat_background_std_threshold: float,
    flat_background_edge_threshold: float,
    context_dilate_px: int,
    context_gap_px: int,
    component_expand_px: int,
    smooth_gradient_edge_threshold: float,
    smooth_gradient_residual_threshold: float,
    smooth_gradient_color_bias_max_delta: float,
    smooth_gradient_color_bias_residual_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int, np.ndarray, np.ndarray]]]:
    component_mask = (mask_array > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    if component_count <= 1:
        return source, mask_array, []

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = np.ones((context_dilate_px * 2 + 1, context_dilate_px * 2 + 1), dtype=np.uint8)
    expand_kernel = None
    if component_expand_px > 0:
        expand_kernel = np.ones((component_expand_px * 2 + 1, component_expand_px * 2 + 1), dtype=np.uint8)
    gap_kernel = None
    if context_gap_px > 0:
        gap_kernel = np.ones((context_gap_px * 2 + 1, context_gap_px * 2 + 1), dtype=np.uint8)

    prefilled = source.astype(np.float32).copy()
    residual_mask = mask_array.copy()
    prefilled_patches: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for component_index, stat in enumerate(stats[1:], start=1):
        _, _, _, _, area = stat
        if area <= 0:
            continue

        component = (labels == component_index).astype(np.uint8)
        expanded_component, patch = _resolve_component_background_patch(
            source,
            gray,
            edges,
            component,
            protected_line_mask=protected_line_mask,
            protected_prefill_ring_mask=protected_prefill_ring_mask,
            kernel=kernel,
            expand_kernel=expand_kernel,
            gap_kernel=gap_kernel,
            flat_background_std_threshold=flat_background_std_threshold,
            flat_background_edge_threshold=flat_background_edge_threshold,
            context_dilate_px=context_dilate_px,
            smooth_gradient_edge_threshold=smooth_gradient_edge_threshold,
            smooth_gradient_residual_threshold=smooth_gradient_residual_threshold,
            smooth_gradient_color_bias_max_delta=smooth_gradient_color_bias_max_delta,
            smooth_gradient_color_bias_residual_scale=smooth_gradient_color_bias_residual_scale,
        )
        if patch is None:
            continue

        x0, y0, x1, y1, patch_values, _ = patch
        local_component = expanded_component[y0:y1, x0:x1].astype(bool)
        if not np.any(local_component):
            continue
        prefilled[y0:y1, x0:x1][local_component] = patch_values[local_component]
        residual_mask[expanded_component.astype(bool)] = 0
        prefilled_patches.append((x0, y0, x1, y1, local_component.copy(), patch_values.copy()))

    return np.clip(prefilled, 0, 255).astype(np.uint8), residual_mask, prefilled_patches


def _inpaint_residual_components(
    source: np.ndarray,
    residual_mask: np.ndarray,
    *,
    protected_line_mask: np.ndarray | None,
    base_radius: float,
    min_radius: float,
    max_radius: float,
    reference_span_px: float,
    edge_density_threshold: float,
    edge_density_min_factor: float,
    small_component_max_span_px: int,
    group_proximity_px: int,
    group_proximity_min_scale: float,
    group_proximity_max_scale: float,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    component_mask = (residual_mask > 0).astype(np.uint8)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    if component_count <= 1:
        return source, []

    repaired = source.copy()
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    component_groups = _build_residual_component_groups(
        labels,
        component_count=component_count,
        small_component_max_span_px=small_component_max_span_px,
        group_proximity_px=group_proximity_px,
        group_proximity_min_scale=group_proximity_min_scale,
        group_proximity_max_scale=group_proximity_max_scale,
    )
    diagnostics: list[dict[str, float | int]] = []
    for group in component_groups:
        component = group["mask"]
        points = cv2.findNonZero(component)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        protected_nearby = False
        if protected_line_mask is not None and np.any(protected_line_mask):
            protected_padding = max(5, int(group["proximity_px"]) + 5)
            protected_kernel = np.ones((protected_padding * 2 + 1, protected_padding * 2 + 1), dtype=np.uint8)
            protected_vicinity = cv2.dilate(component, protected_kernel, iterations=1) > 0
            protected_nearby = bool(np.any(protected_line_mask & protected_vicinity))
        ring_mask = _build_component_ring_mask(component, padding_px=max(2, int(group["proximity_px"])))
        if protected_nearby and protected_line_mask is not None:
            ring_mask &= ~protected_line_mask
        ring_pixel_count = int(np.count_nonzero(ring_mask))
        edge_density = 0.0
        if ring_pixel_count > 0:
            edge_density = float(np.count_nonzero(edges[ring_mask])) / float(ring_pixel_count)
        component_radius = _resolve_component_telea_radius(
            width=width,
            height=height,
            base_radius=base_radius,
            min_radius=min_radius,
            max_radius=max_radius,
            reference_span_px=reference_span_px,
            edge_density=edge_density,
            edge_density_threshold=edge_density_threshold,
            edge_density_min_factor=edge_density_min_factor,
        )
        component_radius = _clamp_isolated_label_telea_radius(
            component_radius,
            width=width,
            height=height,
            edge_density=edge_density,
            group_size=int(group["group_size"]),
            proximity_px=int(group["proximity_px"]),
        )
        if protected_nearby:
            component_radius = max(min_radius, component_radius * 0.65)
            padding_scale = 1.1
        else:
            padding_scale = 2.0
        padding = max(2, int(math.ceil(component_radius * padding_scale)))
        if protected_nearby and protected_line_mask is not None:
            x0, y0, x1, y1 = _resolve_protected_inpaint_crop_bounds(
                protected_line_mask,
                component_bbox=(x, y, width, height),
                image_shape=source.shape[:2],
                base_padding=padding,
            )
        elif _is_compact_wide_residual_component(width=width, height=height, edge_density=edge_density):
            x0, y0, x1, y1 = _resolve_directional_inpaint_crop_bounds(
                gray,
                edges,
                component_bbox=(x, y, width, height),
                image_shape=source.shape[:2],
                base_padding=padding,
            )
        else:
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(source.shape[1], x + width + padding)
            y1 = min(source.shape[0], y + height + padding)

        local_source = repaired[y0:y1, x0:x1]
        local_mask = (component[y0:y1, x0:x1] * 255).astype(np.uint8)
        local_repaired = cv2.inpaint(local_source, local_mask, component_radius, cv2.INPAINT_TELEA)
        local_component = local_mask > 0
        local_target = repaired[y0:y1, x0:x1]
        local_target[local_component] = local_repaired[local_component]
        diagnostics.append(
            {
                "group_size": int(group["group_size"]),
                "proximity_px": int(group["proximity_px"]),
                "edge_density": float(edge_density),
                "final_radius": float(component_radius),
                "protected_nearby": int(protected_nearby),
            }
        )

    return repaired, diagnostics


def _build_residual_component_groups(
    labels: np.ndarray,
    *,
    component_count: int,
    small_component_max_span_px: int,
    group_proximity_px: int,
    group_proximity_min_scale: float,
    group_proximity_max_scale: float,
) -> list[dict[str, np.ndarray | int]]:
    small_components: list[dict[str, np.ndarray | tuple[int, int, int, int] | int]] = []
    groups: list[dict[str, np.ndarray | int]] = []
    for component_index in range(1, component_count):
        component = (labels == component_index).astype(np.uint8)
        points = cv2.findNonZero(component)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        span = max(width, height)
        if span <= small_component_max_span_px:
            small_components.append({
                "mask": component,
                "bbox": (x, y, width, height),
                "span": span,
            })
        else:
            groups.append({"mask": component, "group_size": 1, "proximity_px": 0})

    if not small_components:
        return groups

    parent = list(range(len(small_components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(small_components):
        left_bbox = left["bbox"]
        left_span = int(left["span"])
        for right_index in range(left_index + 1, len(small_components)):
            right = small_components[right_index]
            right_bbox = right["bbox"]
            right_span = int(right["span"])
            adaptive_proximity = _resolve_group_proximity_px(
                base_proximity_px=group_proximity_px,
                span=max(left_span, right_span),
                reference_span_px=small_component_max_span_px,
                min_scale=group_proximity_min_scale,
                max_scale=group_proximity_max_scale,
            )
            if _bbox_gap_px(left_bbox, right_bbox) <= adaptive_proximity:
                union(left_index, right_index)

    grouped_components: dict[int, list[dict[str, np.ndarray | tuple[int, int, int, int] | int]]] = {}
    for index, component in enumerate(small_components):
        grouped_components.setdefault(find(index), []).append(component)

    for members in grouped_components.values():
        group_mask = np.zeros_like(labels, dtype=np.uint8)
        max_span = 0
        for member in members:
            group_mask = np.maximum(group_mask, member["mask"])
            max_span = max(max_span, int(member["span"]))
        groups.append(
            {
                "mask": group_mask,
                "group_size": len(members),
                "proximity_px": _resolve_group_proximity_px(
                    base_proximity_px=group_proximity_px,
                    span=max_span,
                    reference_span_px=small_component_max_span_px,
                    min_scale=group_proximity_min_scale,
                    max_scale=group_proximity_max_scale,
                ),
            }
        )

    return groups


def _resolve_group_proximity_px(
    *,
    base_proximity_px: int,
    span: int,
    reference_span_px: int,
    min_scale: float,
    max_scale: float,
) -> int:
    if base_proximity_px <= 0:
        return 0
    scale = float(np.clip(span / max(1.0, float(reference_span_px)), min_scale, max_scale))
    return max(1, int(round(base_proximity_px * scale)))


def _bbox_gap_px(
    left_bbox: tuple[int, int, int, int] | np.ndarray,
    right_bbox: tuple[int, int, int, int] | np.ndarray,
) -> int:
    left_x, left_y, left_width, left_height = left_bbox
    right_x, right_y, right_width, right_height = right_bbox
    left_right = left_x + left_width
    left_bottom = left_y + left_height
    right_right = right_x + right_width
    right_bottom = right_y + right_height
    x_gap = max(0, right_x - left_right, left_x - right_right)
    y_gap = max(0, right_y - left_bottom, left_y - right_bottom)
    return max(x_gap, y_gap)


def _build_component_ring_mask(component: np.ndarray, *, padding_px: int) -> np.ndarray:
    kernel = np.ones((padding_px * 2 + 1, padding_px * 2 + 1), dtype=np.uint8)
    outer = cv2.dilate(component, kernel, iterations=1).astype(bool)
    inner = component.astype(bool)
    return outer & (~inner)


def _resolve_component_telea_radius(
    *,
    width: int,
    height: int,
    base_radius: float,
    min_radius: float,
    max_radius: float,
    reference_span_px: float,
    edge_density: float,
    edge_density_threshold: float,
    edge_density_min_factor: float,
) -> float:
    component_span = _resolve_component_telea_effective_span(
        width=width,
        height=height,
        edge_density=edge_density,
    )
    size_scale = component_span / reference_span_px
    density_ratio = min(1.0, max(0.0, edge_density) / edge_density_threshold)
    edge_scale = 1.0 - density_ratio * (1.0 - edge_density_min_factor)
    return float(np.clip(base_radius * size_scale * edge_scale, min_radius, max_radius))


def _resolve_component_telea_effective_span(*, width: int, height: int, edge_density: float) -> float:
    span = float(max(width, height))
    if height <= 0:
        return span
    if _is_compact_wide_residual_component(width=width, height=height, edge_density=edge_density):
        return float(math.sqrt(float(width) * float(height)))
    return span


def _is_compact_wide_residual_component(*, width: int, height: int, edge_density: float) -> bool:
    if height <= 0:
        return False
    aspect_ratio = float(width) / float(height)
    return (
        width <= DEFAULT_TELEA_COMPACT_WIDE_MAX_WIDTH_PX
        and height <= DEFAULT_TELEA_COMPACT_WIDE_MAX_HEIGHT_PX
        and aspect_ratio >= DEFAULT_TELEA_COMPACT_WIDE_MIN_ASPECT_RATIO
        and edge_density <= DEFAULT_TELEA_COMPACT_WIDE_EDGE_DENSITY_THRESHOLD
    )


def _clamp_isolated_label_telea_radius(
    radius: float,
    *,
    width: int,
    height: int,
    edge_density: float,
    group_size: int,
    proximity_px: int,
) -> float:
    if height <= 0:
        return radius
    aspect_ratio = float(width) / float(height)
    if group_size != 1 or proximity_px != 0:
        return radius
    if width > DEFAULT_TELEA_ISOLATED_LABEL_MAX_WIDTH_PX:
        return radius
    if height > DEFAULT_TELEA_ISOLATED_LABEL_MAX_HEIGHT_PX:
        return radius
    if aspect_ratio < DEFAULT_TELEA_ISOLATED_LABEL_MIN_ASPECT_RATIO:
        return radius
    if edge_density > DEFAULT_TELEA_ISOLATED_LABEL_EDGE_DENSITY_THRESHOLD:
        return radius
    return min(radius, DEFAULT_TELEA_ISOLATED_LABEL_RADIUS_CAP)


def _resolve_directional_inpaint_crop_bounds(
    gray: np.ndarray,
    edges: np.ndarray,
    *,
    component_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    base_padding: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = component_bbox
    image_height, image_width = image_shape
    x0 = max(0, x - base_padding)
    y0 = max(0, y - base_padding)
    x1 = min(image_width, x + width + base_padding)
    y1 = min(image_height, y + height + base_padding)

    left_region = gray[y:y + height, x0:x]
    right_region = gray[y:y + height, x + width:x1]
    top_region = gray[y0:y, x:x + width]
    bottom_region = gray[y + height:y1, x:x + width]
    left_edges = edges[y:y + height, x0:x]
    right_edges = edges[y:y + height, x + width:x1]
    top_edges = edges[y0:y, x:x + width]
    bottom_edges = edges[y + height:y1, x:x + width]

    vertical_luma = _region_percentile_mean(top_region, bottom_region, percentile=10.0)
    vertical_edge = _region_edge_density_mean(top_edges, bottom_edges)
    ultra_near_padding = 0
    very_near_padding = max(1, int(math.ceil(base_padding * 0.18)))
    near_padding = max(2, int(math.ceil(base_padding * 0.45)))
    far_padding = max(base_padding, int(math.ceil(base_padding * 1.15)))

    left_padding = _resolve_directional_side_padding(
        left_region,
        left_edges,
        vertical_luma=vertical_luma,
        vertical_edge=vertical_edge,
        ultra_near_padding=ultra_near_padding,
        very_near_padding=very_near_padding,
        near_padding=near_padding,
        far_padding=far_padding,
    )
    right_padding = _resolve_directional_side_padding(
        right_region,
        right_edges,
        vertical_luma=vertical_luma,
        vertical_edge=vertical_edge,
        ultra_near_padding=ultra_near_padding,
        very_near_padding=very_near_padding,
        near_padding=near_padding,
        far_padding=far_padding,
    )
    top_padding = far_padding
    bottom_padding = far_padding

    return (
        max(0, x - left_padding),
        max(0, y - top_padding),
        min(image_width, x + width + right_padding),
        min(image_height, y + height + bottom_padding),
    )


def _resolve_directional_side_padding(
    region: np.ndarray,
    edge_region: np.ndarray,
    *,
    vertical_luma: float,
    vertical_edge: float,
    ultra_near_padding: int,
    very_near_padding: int,
    near_padding: int,
    far_padding: int,
) -> int:
    if region.size == 0:
        return very_near_padding
    side_luma = float(np.percentile(region, 10))
    side_edge = float(np.count_nonzero(edge_region)) / float(max(1, region.size))
    if side_luma + DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_LUMA_MARGIN < vertical_luma:
        return ultra_near_padding
    if side_edge > vertical_edge + DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_EDGE_MARGIN:
        return ultra_near_padding
    if side_luma + DEFAULT_TELEA_DIRECTIONAL_SIDE_LUMA_MARGIN < vertical_luma:
        return near_padding
    if side_edge > vertical_edge + DEFAULT_TELEA_DIRECTIONAL_SIDE_EDGE_MARGIN:
        return near_padding
    return far_padding


def _region_percentile_mean(first: np.ndarray, second: np.ndarray, *, percentile: float) -> float:
    values: list[float] = []
    if first.size > 0:
        values.append(float(np.percentile(first, percentile)))
    if second.size > 0:
        values.append(float(np.percentile(second, percentile)))
    if not values:
        return 255.0
    return float(np.mean(values))


def _region_edge_density_mean(first: np.ndarray, second: np.ndarray) -> float:
    values: list[float] = []
    for region in (first, second):
        if region.size > 0:
            values.append(float(np.count_nonzero(region)) / float(region.size))
    if not values:
        return 0.0
    return float(np.mean(values))


def _resolve_protected_inpaint_crop_bounds(
    protected_line_mask: np.ndarray,
    *,
    component_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    base_padding: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = component_bbox
    image_height, image_width = image_shape
    score_padding = max(base_padding, 8)
    symmetric_x0 = max(0, x - score_padding)
    symmetric_y0 = max(0, y - score_padding)
    symmetric_x1 = min(image_width, x + width + score_padding)
    symmetric_y1 = min(image_height, y + height + score_padding)
    near_padding = max(2, int(math.ceil(base_padding * 0.35)))
    far_padding = max(near_padding + 1, base_padding)

    left_score = int(np.count_nonzero(protected_line_mask[symmetric_y0:symmetric_y1, symmetric_x0:x]))
    right_score = int(np.count_nonzero(protected_line_mask[symmetric_y0:symmetric_y1, x + width : symmetric_x1]))
    top_score = int(np.count_nonzero(protected_line_mask[symmetric_y0:y, symmetric_x0:symmetric_x1]))
    bottom_score = int(np.count_nonzero(protected_line_mask[y + height : symmetric_y1, symmetric_x0:symmetric_x1]))

    left_padding = near_padding if left_score > 0 else far_padding
    right_padding = near_padding if right_score > 0 else far_padding
    top_padding = near_padding if top_score > 0 else far_padding
    bottom_padding = near_padding if bottom_score > 0 else far_padding

    x0 = max(0, x - left_padding)
    y0 = max(0, y - top_padding)
    x1 = min(image_width, x + width + right_padding)
    y1 = min(image_height, y + height + bottom_padding)
    return x0, y0, x1, y1


def _format_telea_group_diagnostics(diagnostics: list[dict[str, float | int]]) -> str:
    if not diagnostics:
        return "Residual Telea groups: 0."
    edge_values = [float(item["edge_density"]) for item in diagnostics]
    group_sizes = [int(item["group_size"]) for item in diagnostics]
    radius_values = [float(item["final_radius"]) for item in diagnostics]
    proximity_values = [int(item["proximity_px"]) for item in diagnostics]
    protected_count = sum(int(item.get("protected_nearby", 0)) for item in diagnostics)
    return (
        f"Residual Telea groups: {len(diagnostics)}; "
        f"group size {min(group_sizes)}-{max(group_sizes)}; "
        f"adaptive proximity {min(proximity_values)}-{max(proximity_values)} px; "
        f"edge density {min(edge_values):.3f}-{max(edge_values):.3f}; "
        f"final radius {min(radius_values):.2f}-{max(radius_values):.2f}; "
        f"protected groups {protected_count}."
    )


def _restore_structural_line_regions(
    source: np.ndarray,
    repaired: np.ndarray,
    mask_array: np.ndarray,
    *,
    padding_px: int = DEFAULT_STRUCTURAL_LINE_PADDING_PX,
    min_kernel_px: int = DEFAULT_STRUCTURAL_LINE_MIN_KERNEL_PX,
) -> tuple[np.ndarray, int]:
    if np.count_nonzero(mask_array) == 0:
        return repaired, 0

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    binary[mask_array > 0] = 0
    horizontal_kernel_len = max(min_kernel_px, source.shape[1] // 30)
    vertical_kernel_len = max(min_kernel_px, source.shape[0] // 20)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
    horizontal_lines = _filter_structural_line_candidates(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel),
        orientation="horizontal",
        min_span_px=horizontal_kernel_len,
    )
    vertical_lines = _filter_structural_line_candidates(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel),
        orientation="vertical",
        min_span_px=vertical_kernel_len,
    )
    line_mask = cv2.bitwise_or(horizontal_lines, vertical_lines) > 0
    if not np.any(line_mask):
        return repaired, 0

    grid_region_mask = _build_grid_region_mask(horizontal_lines, vertical_lines, mask_array)
    if not np.any(grid_region_mask):
        return repaired, 0

    result = repaired.copy()
    restored_pixels = 0
    restored_pixels += _bridge_horizontal_line_gaps(
        source,
        result,
        horizontal_lines,
        mask_array,
        grid_region_mask,
        search_radius=max(horizontal_kernel_len, padding_px * 4),
    )
    restored_pixels += _bridge_vertical_line_gaps(
        source,
        result,
        vertical_lines,
        mask_array,
        grid_region_mask,
        search_radius=max(vertical_kernel_len, padding_px * 4),
    )
    if restored_pixels == 0:
        return repaired, 0
    return result, restored_pixels


def _bridge_horizontal_line_gaps(
    source: np.ndarray,
    result: np.ndarray,
    horizontal_lines: np.ndarray,
    mask_array: np.ndarray,
    grid_region_mask: np.ndarray,
    *,
    search_radius: int,
) -> int:
    restored_pixels = 0
    height, width = mask_array.shape
    for row in range(height):
        if not np.any(horizontal_lines[row] > 0) or not np.any(mask_array[row] > 0):
            continue
        for start, end in _iter_true_runs(mask_array[row] > 0):
            if not np.any(grid_region_mask[row, start : end + 1]):
                continue
            left_slice = slice(max(0, start - search_radius), start)
            right_slice = slice(end + 1, min(width, end + 1 + search_radius))
            left_support = np.flatnonzero(horizontal_lines[row, left_slice] > 0)
            right_support = np.flatnonzero(horizontal_lines[row, right_slice] > 0)
            if left_support.size == 0 or right_support.size == 0:
                continue
            left_positions = left_support + left_slice.start
            right_positions = right_support + right_slice.start
            support_values = np.concatenate(
                [source[row, left_positions].astype(np.float32), source[row, right_positions].astype(np.float32)],
                axis=0,
            )
            fill_color = np.mean(support_values, axis=0)
            result[row, start : end + 1][mask_array[row, start : end + 1] > 0] = fill_color
            restored_pixels += end - start + 1
    return restored_pixels


def _bridge_vertical_line_gaps(
    source: np.ndarray,
    result: np.ndarray,
    vertical_lines: np.ndarray,
    mask_array: np.ndarray,
    grid_region_mask: np.ndarray,
    *,
    search_radius: int,
) -> int:
    restored_pixels = 0
    height, width = mask_array.shape
    for col in range(width):
        if not np.any(vertical_lines[:, col] > 0) or not np.any(mask_array[:, col] > 0):
            continue
        for start, end in _iter_true_runs(mask_array[:, col] > 0):
            if not np.any(grid_region_mask[start : end + 1, col]):
                continue
            top_slice = slice(max(0, start - search_radius), start)
            bottom_slice = slice(end + 1, min(height, end + 1 + search_radius))
            top_support = np.flatnonzero(vertical_lines[top_slice, col] > 0)
            bottom_support = np.flatnonzero(vertical_lines[bottom_slice, col] > 0)
            if top_support.size == 0 or bottom_support.size == 0:
                continue
            top_positions = top_support + top_slice.start
            bottom_positions = bottom_support + bottom_slice.start
            support_values = np.concatenate(
                [source[top_positions, col].astype(np.float32), source[bottom_positions, col].astype(np.float32)],
                axis=0,
            )
            fill_color = np.mean(support_values, axis=0)
            result[start : end + 1, col][mask_array[start : end + 1, col] > 0] = fill_color
            restored_pixels += end - start + 1
    return restored_pixels


def _iter_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _build_grid_region_mask(
    horizontal_lines: np.ndarray,
    vertical_lines: np.ndarray,
    mask_array: np.ndarray,
) -> np.ndarray:
    horizontal_bool = horizontal_lines > 0
    vertical_bool = vertical_lines > 0
    if not np.any(horizontal_bool) or not np.any(vertical_bool):
        return np.zeros_like(mask_array, dtype=bool)

    intersection_seed = horizontal_bool & vertical_bool
    if not np.any(intersection_seed):
        horizontal_kernel = np.ones((1, max(3, mask_array.shape[1] // 80)), dtype=np.uint8)
        vertical_kernel = np.ones((max(3, mask_array.shape[0] // 80), 1), dtype=np.uint8)
        expanded_horizontal = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1) > 0
        expanded_vertical = cv2.dilate(vertical_lines, vertical_kernel, iterations=1) > 0
        intersection_seed = expanded_horizontal & expanded_vertical
        if not np.any(intersection_seed):
            return np.zeros_like(mask_array, dtype=bool)

    region_kernel = np.ones((max(5, mask_array.shape[0] // 40), max(5, mask_array.shape[1] // 40)), dtype=np.uint8)
    grid_region = cv2.dilate(intersection_seed.astype(np.uint8), region_kernel, iterations=1) > 0
    line_region = cv2.dilate((horizontal_bool | vertical_bool).astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
    mask_vicinity = cv2.dilate((mask_array > 0).astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1) > 0
    return grid_region & line_region & mask_vicinity


def _filter_structural_line_candidates(
    line_mask: np.ndarray,
    *,
    orientation: str,
    min_span_px: int,
) -> np.ndarray:
    component_mask = (line_mask > 0).astype(np.uint8)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    if component_count <= 1:
        return line_mask

    filtered = np.zeros_like(line_mask)
    for component_index in range(1, component_count):
        component = (labels == component_index).astype(np.uint8)
        points = cv2.findNonZero(component)
        if points is None:
            continue
        _, _, width, height = cv2.boundingRect(points)
        if orientation == "horizontal":
            if width < min_span_px or width < height * 4:
                continue
        else:
            if height < min_span_px or height < width * 4:
                continue
        filtered[component.astype(bool)] = 255

    return filtered


def _restore_low_texture_regions(
    source: np.ndarray,
    repaired: np.ndarray,
    mask_array: np.ndarray,
    *,
    prefilled_patches: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] | None,
    protected_line_mask: np.ndarray | None,
    flat_background_std_threshold: float,
    flat_background_edge_threshold: float,
    context_dilate_px: int,
    context_gap_px: int,
    component_expand_px: int,
    smooth_gradient_edge_threshold: float,
    smooth_gradient_residual_threshold: float,
    smooth_gradient_color_bias_max_delta: float,
    smooth_gradient_color_bias_residual_scale: float,
    blend_sigma: float,
) -> np.ndarray:
    if prefilled_patches:
        blended = repaired.astype(np.float32)
        for x0, y0, x1, y1, local_component, patch_values in prefilled_patches:
            alpha = cv2.GaussianBlur(local_component.astype(np.float32), (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma)
            alpha = np.clip(alpha * 1.5, 0.0, 1.0)[..., None]
            blended[y0:y1, x0:x1] = blended[y0:y1, x0:x1] * (1.0 - alpha) + patch_values * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    component_mask = (mask_array > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    if component_count <= 1:
        return repaired

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = np.ones((context_dilate_px * 2 + 1, context_dilate_px * 2 + 1), dtype=np.uint8)
    expand_kernel = None
    if component_expand_px > 0:
        expand_kernel = np.ones((component_expand_px * 2 + 1, component_expand_px * 2 + 1), dtype=np.uint8)
    gap_kernel = None
    if context_gap_px > 0:
        gap_kernel = np.ones((context_gap_px * 2 + 1, context_gap_px * 2 + 1), dtype=np.uint8)
    blended = repaired.astype(np.float32)
    protected_prefill_ring_mask = None
    if protected_line_mask is not None:
        protected_prefill_ring_mask = np.zeros(mask_array.shape, dtype=np.uint8)

    for component_index, stat in enumerate(stats[1:], start=1):
        _, _, _, _, area = stat
        if area <= 0:
            continue

        component = (labels == component_index).astype(np.uint8)
        expanded_component, patch = _resolve_component_background_patch(
            source,
            gray,
            edges,
            component,
            protected_line_mask=protected_line_mask,
            protected_prefill_ring_mask=protected_prefill_ring_mask,
            kernel=kernel,
            expand_kernel=expand_kernel,
            gap_kernel=gap_kernel,
            flat_background_std_threshold=flat_background_std_threshold,
            flat_background_edge_threshold=flat_background_edge_threshold,
            context_dilate_px=context_dilate_px,
            smooth_gradient_edge_threshold=smooth_gradient_edge_threshold,
            smooth_gradient_residual_threshold=smooth_gradient_residual_threshold,
            smooth_gradient_color_bias_max_delta=smooth_gradient_color_bias_max_delta,
            smooth_gradient_color_bias_residual_scale=smooth_gradient_color_bias_residual_scale,
        )
        if patch is None:
            continue

        x0, y0, x1, y1, patch_values, _ = patch
        local_component = expanded_component[y0:y1, x0:x1].astype(np.float32)
        alpha = cv2.GaussianBlur(local_component, (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma)
        alpha = np.clip(alpha * 1.5, 0.0, 1.0)[..., None]
        blended[y0:y1, x0:x1] = (
            blended[y0:y1, x0:x1] * (1.0 - alpha) + patch_values * alpha
        )

    return np.clip(blended, 0, 255).astype(np.uint8)


def _resolve_component_background_patch(
    source: np.ndarray,
    gray: np.ndarray,
    edges: np.ndarray,
    component: np.ndarray,
    *,
    protected_line_mask: np.ndarray | None,
    protected_prefill_ring_mask: np.ndarray | None,
    kernel: np.ndarray,
    expand_kernel: np.ndarray | None,
    gap_kernel: np.ndarray | None,
    flat_background_std_threshold: float,
    flat_background_edge_threshold: float,
    context_dilate_px: int,
    smooth_gradient_edge_threshold: float,
    smooth_gradient_residual_threshold: float,
    smooth_gradient_color_bias_max_delta: float,
    smooth_gradient_color_bias_residual_scale: float,
) -> tuple[np.ndarray, tuple[int, int, int, int, np.ndarray, float] | None]:
    expanded_component = component
    if expand_kernel is not None:
        expanded_component = cv2.dilate(component, expand_kernel, iterations=1)

    context_component = expanded_component
    if gap_kernel is not None:
        context_component = cv2.dilate(expanded_component, gap_kernel, iterations=1)

    ring_mask = cv2.dilate(context_component, kernel, iterations=1).astype(bool) & (~context_component.astype(bool))
    if protected_line_mask is not None and np.any(protected_line_mask):
        protected_vicinity = cv2.dilate(component, np.ones((11, 11), dtype=np.uint8), iterations=1) > 0
        if np.any(protected_line_mask & protected_vicinity):
            ring_mask &= ~protected_line_mask
            if protected_prefill_ring_mask is not None:
                protected_prefill_ring_mask[ring_mask] = 255
    ring_pixel_count = int(np.count_nonzero(ring_mask))
    if ring_pixel_count < 64:
        return expanded_component, None

    context_values = gray[ring_mask]
    luma_std = float(np.std(context_values))
    edge_density = float(np.count_nonzero(edges[ring_mask])) / float(ring_pixel_count)
    component_points = cv2.findNonZero(expanded_component)
    if component_points is None:
        return expanded_component, None
    _, _, component_width, component_height = cv2.boundingRect(component_points)
    patch = None
    patch_model: str | None = None
    if luma_std <= flat_background_std_threshold and edge_density <= flat_background_edge_threshold:
        patch = _fit_component_background_surface(
            source,
            expanded_component,
            ring_mask,
            context_dilate_px=context_dilate_px,
            model="plane",
        )
        patch_model = "plane"
    elif edge_density <= smooth_gradient_edge_threshold:
        patch = _fit_component_background_surface(
            source,
            expanded_component,
            ring_mask,
            context_dilate_px=context_dilate_px,
            model="quadratic",
        )
        if patch is not None and patch[-1] > smooth_gradient_residual_threshold:
            patch = None
        else:
            patch_model = "quadratic"
    elif _should_try_relaxed_quadratic_prefill(
        width=component_width,
        height=component_height,
        edge_density=edge_density,
    ):
        patch = _fit_component_background_surface(
            source,
            expanded_component,
            ring_mask,
            context_dilate_px=context_dilate_px,
            model="quadratic",
        )
        if patch is not None and patch[-1] > smooth_gradient_residual_threshold:
            patch = None
        else:
            patch_model = "quadratic-relaxed"

    if patch is None:
        return expanded_component, None

    x0, y0, x1, y1, patch_values, patch_residual = patch
    color_bias_max_delta = smooth_gradient_color_bias_max_delta
    if patch_model == "quadratic-relaxed" and color_bias_max_delta <= 0.0:
        color_bias_max_delta = DEFAULT_RELAXED_QUADRATIC_COLOR_BIAS_MAX_DELTA

    if patch_model in {"quadratic", "quadratic-relaxed"} and color_bias_max_delta > 0.0:
        local_ring = ring_mask[y0:y1, x0:x1]
        source_patch = source[y0:y1, x0:x1]
        local_component_mask = expanded_component[y0:y1, x0:x1].astype(bool)
        effective_max_delta = color_bias_max_delta * min(
            1.0,
            patch_residual / smooth_gradient_color_bias_residual_scale,
        )
        patch_values = _align_patch_mean_to_ring_background(
            patch_values,
            source_patch,
            local_component_mask,
            local_ring,
            max_delta=effective_max_delta,
        )
        patch = (x0, y0, x1, y1, patch_values, patch_residual)
    return expanded_component, patch


def _should_try_relaxed_quadratic_prefill(*, width: int, height: int, edge_density: float) -> bool:
    if width < DEFAULT_RELAXED_QUADRATIC_MIN_WIDTH_PX:
        return False
    if height > DEFAULT_RELAXED_QUADRATIC_MAX_HEIGHT_PX:
        return False
    if width / max(1.0, float(height)) < DEFAULT_RELAXED_QUADRATIC_MIN_ASPECT_RATIO:
        return False
    return edge_density <= DEFAULT_RELAXED_QUADRATIC_EDGE_THRESHOLD


def _fit_component_background_surface(
    source: np.ndarray,
    component: np.ndarray,
    ring_mask: np.ndarray,
    *,
    context_dilate_px: int,
    model: str,
) -> tuple[int, int, int, int, np.ndarray, float] | None:
    points = cv2.findNonZero(component)
    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(points)
    image_height, image_width = source.shape[:2]
    padding = context_dilate_px * 2
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image_width, x + width + padding)
    y1 = min(image_height, y + height + padding)

    local_ring = ring_mask[y0:y1, x0:x1]
    ring_y, ring_x = np.nonzero(local_ring)
    if ring_x.size < 32:
        return None

    design_matrix = _build_surface_design_matrix(ring_x.astype(np.float32), ring_y.astype(np.float32), model=model)
    patch_height = y1 - y0
    patch_width = x1 - x0
    grid_y, grid_x = np.indices((patch_height, patch_width), dtype=np.float32)
    patch_design_matrix = _build_surface_design_matrix(grid_x.reshape(-1), grid_y.reshape(-1), model=model)
    patch_values = np.empty((patch_height, patch_width, source.shape[2]), dtype=np.float32)
    source_patch = source[y0:y1, x0:x1]
    residuals: list[float] = []
    for channel_index in range(source.shape[2]):
        channel_values = source_patch[:, :, channel_index][local_ring].astype(np.float32)
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, channel_values, rcond=None)
        fitted_ring = design_matrix @ coefficients
        residuals.append(float(np.mean(np.abs(fitted_ring - channel_values))))
        patch_values[:, :, channel_index] = (patch_design_matrix @ coefficients).reshape(patch_height, patch_width)

    return x0, y0, x1, y1, np.clip(patch_values, 0.0, 255.0), float(np.mean(residuals))


def _build_surface_design_matrix(x: np.ndarray, y: np.ndarray, *, model: str) -> np.ndarray:
    if model == "plane":
        return np.stack(
            [
                x,
                y,
                np.ones_like(x, dtype=np.float32),
            ],
            axis=1,
        )

    if model == "quadratic":
        return np.stack(
            [
                x * x,
                y * y,
                x * y,
                x,
                y,
                np.ones_like(x, dtype=np.float32),
            ],
            axis=1,
        )

    raise ValueError(f"Unsupported surface model: {model}")


def _align_patch_mean_to_ring_background(
    patch_values: np.ndarray,
    source_patch: np.ndarray,
    local_component_mask: np.ndarray,
    local_ring: np.ndarray,
    *,
    max_delta: float,
) -> np.ndarray:
    if max_delta <= 0.0 or not np.any(local_ring) or not np.any(local_component_mask):
        return patch_values

    corrected = patch_values.copy()
    for channel_index in range(patch_values.shape[2]):
        ring_values = source_patch[:, :, channel_index][local_ring].astype(np.float32)
        component_values = patch_values[:, :, channel_index][local_component_mask].astype(np.float32)
        if ring_values.size == 0 or component_values.size == 0:
            continue
        ring_values.sort()
        trim = ring_values.size // 10
        if trim > 0 and ring_values.size > trim * 2:
            ring_values = ring_values[trim:-trim]
        ring_mean = float(np.mean(ring_values))
        component_mean = float(np.mean(component_values))
        delta = ring_mean - component_mean
        if abs(delta) < 1e-3:
            continue
        corrected[:, :, channel_index] = np.clip(
            corrected[:, :, channel_index] + np.clip(delta, -max_delta, max_delta),
            0.0,
            255.0,
        )

    return corrected


__all__ = [
    "BackgroundInpaintingEngine",
    "BackgroundInpaintingError",
    "BackgroundRenderResult",
    "LamaOnnxCudaInpaintingEngine",
    "LamaPytorchInpaintingEngine",
    "OpenCvFastInpaintingEngine",
    "WhiteBoxInpaintingEngine",
]
