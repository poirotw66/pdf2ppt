"""Shared base types, exceptions, and tuning constants for background inpainting engines.

This module is the Phase 1.3 split's foundation: the engine base class, the result/exception
types, and every ``DEFAULT_*`` tuning constant used by the other engine submodules
(``white_box``, ``opencv_fast``, ``lama``, ``patching``, ``compositing``) live here so that
every submodule can import them from a single, dependency-free place. This is a pure move out
of the former monolithic ``inpainting_engines.py`` -- no defaults, thresholds, or behavior were
changed while splitting the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

# --- OpenCvFastInpaintingEngine tuning constants (used by opencv_fast.py) -------------------
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

# --- LaMa (ONNX + PyTorch) tuning constants (used by lama.py, patching.py, compositing.py) --
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
# "lama-onnx" is the canonical CPU/GPU-capable ONNX engine identifier (Phase 1.1).
# "lama-onnx-cuda" (and its hybrid variant) are kept as backward-compatible aliases so
# existing CLI invocations and API requests referencing the old GPU-only name keep working.
_LAMA_ONNX_LEGACY_ALIASES = {
    "lama-onnx-cuda": "lama-onnx",
    "lama-onnx-cuda-hybrid": "lama-onnx-hybrid",
}
LAMA_INPAINT_ENGINES = frozenset(
    {
        "lama-pytorch",
        "lama-onnx",
        "lama-onnx-cuda",
        "lama-pytorch-hybrid",
        "lama-onnx-hybrid",
        "lama-onnx-cuda-hybrid",
    }
)
LAMA_HYBRID_INPAINT_ENGINES = frozenset(
    {
        "lama-pytorch-hybrid",
        "lama-onnx-hybrid",
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


def base_lama_inpaint_engine(engine: str) -> str:
    normalized = _LAMA_ONNX_LEGACY_ALIASES.get(engine, engine)
    if normalized == "lama-pytorch-hybrid":
        return "lama-pytorch"
    if normalized == "lama-onnx-hybrid":
        return "lama-onnx"
    return normalized


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


__all__ = [
    "DEFAULT_LAMA_COMPOSITE_ALPHA_GAIN",
    "DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA",
    "DEFAULT_LAMA_COMPOSITE_HARD_CORE_ERODE_PX",
    "DEFAULT_LAMA_COMPOSITE_MASK_DILATE_PX",
    "DEFAULT_LAMA_EDGE_FEATHER_PX",
    "DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX",
    "DEFAULT_LAMA_MASK_CLOSE_PX",
    "DEFAULT_LAMA_MASK_EXTRA_PADDING_PX",
    "DEFAULT_LAMA_ONNX_MAX_SIDE_PX",
    "DEFAULT_LAMA_ONNX_MODEL_FILENAMES",
    "DEFAULT_LAMA_PATCH_CONTEXT_PADDING_PX",
    "DEFAULT_LAMA_PATCH_HYBRID_COMPLEXITY_THRESHOLD",
    "DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD",
    "DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD",
    "DEFAULT_LAMA_PATCH_MAX_COMPONENTS_FOR_FULL_PAGE",
    "DEFAULT_LAMA_PATCH_MAX_SIDE_PX",
    "DEFAULT_LAMA_PATCH_MERGE_GAP_PX",
    "DEFAULT_RELAXED_QUADRATIC_COLOR_BIAS_MAX_DELTA",
    "DEFAULT_RELAXED_QUADRATIC_EDGE_THRESHOLD",
    "DEFAULT_RELAXED_QUADRATIC_MAX_HEIGHT_PX",
    "DEFAULT_RELAXED_QUADRATIC_MIN_ASPECT_RATIO",
    "DEFAULT_RELAXED_QUADRATIC_MIN_WIDTH_PX",
    "DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_MAX_DELTA",
    "DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_RESIDUAL_SCALE",
    "DEFAULT_SMOOTH_GRADIENT_CONTEXT_GAP_PX",
    "DEFAULT_SMOOTH_GRADIENT_EDGE_THRESHOLD",
    "DEFAULT_SMOOTH_GRADIENT_QUADRATIC_RESIDUAL_THRESHOLD",
    "DEFAULT_STRUCTURAL_LINE_MIN_KERNEL_PX",
    "DEFAULT_STRUCTURAL_LINE_PADDING_PX",
    "DEFAULT_TELEA_COMPACT_WIDE_EDGE_DENSITY_THRESHOLD",
    "DEFAULT_TELEA_COMPACT_WIDE_MAX_HEIGHT_PX",
    "DEFAULT_TELEA_COMPACT_WIDE_MAX_WIDTH_PX",
    "DEFAULT_TELEA_COMPACT_WIDE_MIN_ASPECT_RATIO",
    "DEFAULT_TELEA_DIRECTIONAL_SIDE_EDGE_MARGIN",
    "DEFAULT_TELEA_DIRECTIONAL_SIDE_LUMA_MARGIN",
    "DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_EDGE_MARGIN",
    "DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_LUMA_MARGIN",
    "DEFAULT_TELEA_GROUP_PROXIMITY_MAX_SCALE",
    "DEFAULT_TELEA_GROUP_PROXIMITY_MIN_SCALE",
    "DEFAULT_TELEA_GROUP_PROXIMITY_PX",
    "DEFAULT_TELEA_ISOLATED_LABEL_EDGE_DENSITY_THRESHOLD",
    "DEFAULT_TELEA_ISOLATED_LABEL_MAX_HEIGHT_PX",
    "DEFAULT_TELEA_ISOLATED_LABEL_MAX_WIDTH_PX",
    "DEFAULT_TELEA_ISOLATED_LABEL_MIN_ASPECT_RATIO",
    "DEFAULT_TELEA_ISOLATED_LABEL_RADIUS_CAP",
    "DEFAULT_TELEA_RADIUS_EDGE_DENSITY_MIN_FACTOR",
    "DEFAULT_TELEA_RADIUS_EDGE_DENSITY_THRESHOLD",
    "DEFAULT_TELEA_RADIUS_MAX_SCALE",
    "DEFAULT_TELEA_RADIUS_MIN_SCALE",
    "DEFAULT_TELEA_RADIUS_REFERENCE_SPAN_PX",
    "DEFAULT_TELEA_SMALL_COMPONENT_MAX_SPAN_PX",
    "LAMA_HYBRID_INPAINT_ENGINES",
    "LAMA_INPAINT_ENGINES",
    "BackgroundInpaintingEngine",
    "BackgroundInpaintingError",
    "BackgroundRenderResult",
    "base_lama_inpaint_engine",
    "uses_lama_inpaint_engine",
    "uses_lama_patch_hybrid",
]
