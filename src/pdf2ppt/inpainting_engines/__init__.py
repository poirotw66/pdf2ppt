"""Background inpainting engines package.

Phase 1.3 split of the former monolithic ``inpainting_engines.py`` (2,605 lines) into focused
submodules, following the structure proposed in ``docs/optimization-roadmap.md``:

- ``base``: engine base class, result/exception types, and every tuning constant.
- ``white_box``: the ``white-box`` engine.
- ``opencv_fast``: the ``opencv-fast`` engine and its prefill/residual/structural-line helpers.
- ``compositing``: alpha-blend compositing of LaMa-restored crops/pages.
- ``patching``: LaMa patch-group splitting and opencv-fast/LaMa hybrid routing.
- ``lama``: the ONNX and PyTorch LaMa engines (the PyTorch subprocess/conda-env management
  stack is kept as-is; Phase 1.2, in-process PyTorch inference, is out of scope for this split).

This ``__init__`` re-exports the complete public *and* private surface of the former single
file so every existing import path (``from pdf2ppt.inpainting_engines import ...``,
``pdf2ppt.inpainting_engines.<name>``, and test mock-patch targets on module-singleton
attributes like ``cv2``/``subprocess``/``importlib``) keeps working unchanged. This was a pure
move: no defaults, thresholds, or algorithmic behavior changed as part of the split.

Note for maintainers/tests: a handful of private helpers that used to live in the *same* module
as their sole caller (e.g. ``_prefill_low_texture_regions``, called from
``OpenCvFastInpaintingEngine.inpaint`` in ``opencv_fast.py``) are now defined in a submodule.
Python resolves a bare-name call against the *defining* module's globals, so
``unittest.mock.patch("pdf2ppt.inpainting_engines.<name>", ...)`` no longer reaches those call
sites -- patch the submodule directly instead, e.g.
``pdf2ppt.inpainting_engines.opencv_fast._prefill_low_texture_regions``. Singleton third-party
modules (``cv2``, ``subprocess``, ``importlib``) are unaffected by this: patching an attribute
on the shared module object (however it was imported) is visible everywhere.
"""

from __future__ import annotations

# Re-exported for backward-compatible mock-patch targets such as
# ``patch("pdf2ppt.inpainting_engines.cv2.inpaint")`` /
# ``patch("pdf2ppt.inpainting_engines.subprocess.run")`` /
# ``patch("pdf2ppt.inpainting_engines.importlib.import_module")``. These patch the shared
# module object's attribute, so it doesn't matter that the actual call sites live in a
# submodule now -- as long as this package itself exposes the same singleton module.
import cv2  # noqa: F401
import importlib  # noqa: F401
import subprocess  # noqa: F401

from .base import (
    DEFAULT_LAMA_COMPOSITE_ALPHA_GAIN,
    DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
    DEFAULT_LAMA_COMPOSITE_HARD_CORE_ERODE_PX,
    DEFAULT_LAMA_COMPOSITE_MASK_DILATE_PX,
    DEFAULT_LAMA_EDGE_FEATHER_PX,
    DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX,
    DEFAULT_LAMA_MASK_CLOSE_PX,
    DEFAULT_LAMA_MASK_EXTRA_PADDING_PX,
    DEFAULT_LAMA_ONNX_MAX_SIDE_PX,
    DEFAULT_LAMA_ONNX_MODEL_FILENAMES,
    DEFAULT_LAMA_PATCH_CONTEXT_PADDING_PX,
    DEFAULT_LAMA_PATCH_HYBRID_COMPLEXITY_THRESHOLD,
    DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD,
    DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD,
    DEFAULT_LAMA_PATCH_MAX_COMPONENTS_FOR_FULL_PAGE,
    DEFAULT_LAMA_PATCH_MAX_SIDE_PX,
    DEFAULT_LAMA_PATCH_MERGE_GAP_PX,
    DEFAULT_RELAXED_QUADRATIC_COLOR_BIAS_MAX_DELTA,
    DEFAULT_RELAXED_QUADRATIC_EDGE_THRESHOLD,
    DEFAULT_RELAXED_QUADRATIC_MAX_HEIGHT_PX,
    DEFAULT_RELAXED_QUADRATIC_MIN_ASPECT_RATIO,
    DEFAULT_RELAXED_QUADRATIC_MIN_WIDTH_PX,
    DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_MAX_DELTA,
    DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_RESIDUAL_SCALE,
    DEFAULT_SMOOTH_GRADIENT_CONTEXT_GAP_PX,
    DEFAULT_SMOOTH_GRADIENT_EDGE_THRESHOLD,
    DEFAULT_SMOOTH_GRADIENT_QUADRATIC_RESIDUAL_THRESHOLD,
    DEFAULT_STRUCTURAL_LINE_MIN_KERNEL_PX,
    DEFAULT_STRUCTURAL_LINE_PADDING_PX,
    DEFAULT_TELEA_COMPACT_WIDE_EDGE_DENSITY_THRESHOLD,
    DEFAULT_TELEA_COMPACT_WIDE_MAX_HEIGHT_PX,
    DEFAULT_TELEA_COMPACT_WIDE_MAX_WIDTH_PX,
    DEFAULT_TELEA_COMPACT_WIDE_MIN_ASPECT_RATIO,
    DEFAULT_TELEA_DIRECTIONAL_SIDE_EDGE_MARGIN,
    DEFAULT_TELEA_DIRECTIONAL_SIDE_LUMA_MARGIN,
    DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_EDGE_MARGIN,
    DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_LUMA_MARGIN,
    DEFAULT_TELEA_GROUP_PROXIMITY_MAX_SCALE,
    DEFAULT_TELEA_GROUP_PROXIMITY_MIN_SCALE,
    DEFAULT_TELEA_GROUP_PROXIMITY_PX,
    DEFAULT_TELEA_ISOLATED_LABEL_EDGE_DENSITY_THRESHOLD,
    DEFAULT_TELEA_ISOLATED_LABEL_MAX_HEIGHT_PX,
    DEFAULT_TELEA_ISOLATED_LABEL_MAX_WIDTH_PX,
    DEFAULT_TELEA_ISOLATED_LABEL_MIN_ASPECT_RATIO,
    DEFAULT_TELEA_ISOLATED_LABEL_RADIUS_CAP,
    DEFAULT_TELEA_RADIUS_EDGE_DENSITY_MIN_FACTOR,
    DEFAULT_TELEA_RADIUS_EDGE_DENSITY_THRESHOLD,
    DEFAULT_TELEA_RADIUS_MAX_SCALE,
    DEFAULT_TELEA_RADIUS_MIN_SCALE,
    DEFAULT_TELEA_RADIUS_REFERENCE_SPAN_PX,
    DEFAULT_TELEA_SMALL_COMPONENT_MAX_SPAN_PX,
    LAMA_HYBRID_INPAINT_ENGINES,
    LAMA_INPAINT_ENGINES,
    BackgroundInpaintingEngine,
    BackgroundInpaintingError,
    BackgroundRenderResult,
    base_lama_inpaint_engine,
    uses_lama_inpaint_engine,
    uses_lama_patch_hybrid,
)
from .compositing import (
    _build_lama_composite_alpha,
    _composite_lama_restoration,
    _composite_lama_restoration_with_inference_mask,
    _expand_lama_inference_mask,
    _finalize_lama_inpaint,
    _lama_composite_debug_suffix,
)
from .lama import (
    LamaOnnxCudaInpaintingEngine,
    LamaPytorchInpaintingEngine,
    _build_lama_model_inputs,
    _build_lama_subprocess_env,
    _coerce_lama_dimension,
    _default_lama_python_candidates,
    _fit_lama_inputs_to_model,
    _get_lama_pytorch_worker,
    _get_lama_session,
    _import_onnxruntime,
    _normalize_lama_output,
    _pad_lama_inputs,
    _read_lama_pytorch_response,
    _resize_lama_inputs,
    _resolve_lama_execution_provider,
    _resolve_lama_fixed_input_size,
    _resolve_lama_model_path,
    _resolve_lama_python_executable,
    _resolve_lama_pytorch_model_path,
    _resolve_lama_repo_root,
    _resolve_lama_session_provider,
    _run_lama_pytorch_prediction,
    _shutdown_lama_pytorch_workers,
    _validate_lama_pytorch_runtime,
)
from .opencv_fast import (
    OpenCvFastInpaintingEngine,
    _align_patch_mean_to_ring_background,
    _bbox_gap_px,
    _bridge_horizontal_line_gaps,
    _bridge_vertical_line_gaps,
    _build_component_ring_mask,
    _build_grid_region_mask,
    _build_residual_component_groups,
    _build_surface_design_matrix,
    _clamp_isolated_label_telea_radius,
    _filter_structural_line_candidates,
    _fit_component_background_surface,
    _format_telea_group_diagnostics,
    _inpaint_residual_components,
    _is_compact_wide_residual_component,
    _iter_true_runs,
    _prefill_low_texture_regions,
    _region_edge_density_mean,
    _region_percentile_mean,
    _resolve_component_background_patch,
    _resolve_component_telea_effective_span,
    _resolve_component_telea_radius,
    _resolve_directional_inpaint_crop_bounds,
    _resolve_directional_side_padding,
    _resolve_group_proximity_px,
    _resolve_protected_inpaint_crop_bounds,
    _restore_low_texture_regions,
    _restore_structural_line_regions,
    _should_try_relaxed_quadratic_prefill,
)
from .patching import (
    _boxes_overlap_with_gap,
    _build_lama_patch_groups,
    _composite_lama_patch_into_page,
    _inpaint_lama_crop_with_hybrid,
    _inpaint_lama_page_by_patches,
    _inpaint_opencv_fast_crop,
    _merge_lama_patch_boxes,
    _prepare_lama_working_mask,
    _should_use_lama_patch_inpaint,
    _should_use_opencv_for_lama_patch,
)
from .white_box import WhiteBoxInpaintingEngine

__all__ = [
    "BackgroundInpaintingEngine",
    "BackgroundInpaintingError",
    "BackgroundRenderResult",
    "LamaOnnxCudaInpaintingEngine",
    "LamaPytorchInpaintingEngine",
    "OpenCvFastInpaintingEngine",
    "WhiteBoxInpaintingEngine",
]
