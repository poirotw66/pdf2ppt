"""LaMa background inpainting engines: ONNX (in-process, CPU/GPU) and PyTorch (subprocess).

``LamaOnnxCudaInpaintingEngine`` runs the ONNX Runtime session in-process, falling back to
``CPUExecutionProvider`` when the requested provider (e.g. a CUDA provider) is unavailable
(Phase 1.1). ``LamaPytorchInpaintingEngine`` drives the official advimman/lama checkpoint
through a persistent subprocess worker in a separate Python environment -- that whole
subprocess/conda management stack is intentionally kept as-is here (Phase 1.2, in-process
PyTorch inference, is out of scope for this split).

Split out of the former monolithic ``inpainting_engines.py`` (Phase 1.3, pure move) -- no
subprocess protocol, caching, or inference behavior was changed while splitting the file.
"""

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
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .base import (
    DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX,
    DEFAULT_LAMA_ONNX_MAX_SIDE_PX,
    DEFAULT_LAMA_ONNX_MODEL_FILENAMES,
    DEFAULT_LAMA_PATCH_MAX_SIDE_PX,
    BackgroundInpaintingEngine,
    BackgroundInpaintingError,
)
from .compositing import _expand_lama_inference_mask, _finalize_lama_inpaint, _lama_composite_debug_suffix
from .patching import (
    _inpaint_lama_page_by_patches,
    _inpaint_opencv_fast_crop,
    _prepare_lama_working_mask,
    _should_use_lama_patch_inpaint,
    _should_use_opencv_for_lama_patch,
)

logger = logging.getLogger(__name__)

_LAMA_SESSION_CACHE: dict[tuple[str, str, str], Any] = {}
_LAMA_SESSION_CACHE_LOCK = Lock()


class LamaOnnxCudaInpaintingEngine(BackgroundInpaintingEngine):
    """ONNX-backed LaMa inpainting engine.

    Despite the class name (kept for backward-compatible import paths), this engine is no
    longer CUDA-only: Phase 1.1 lets it fall back to ``CPUExecutionProvider`` when the
    requested provider is unavailable, so ``name`` is the CPU/GPU-neutral ``"lama-onnx"``.
    ``"lama-onnx-cuda"`` remains a valid engine identifier via ``base_lama_inpaint_engine``.
    """

    name = "lama-onnx"

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
                f"LaMa ONNX provider={_resolve_lama_session_provider(session, self.cuda_provider)} "
            f"requested-provider={self.cuda_provider} execution_mode={self.execution_mode} "
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
            f"LaMa ONNX provider={_resolve_lama_session_provider(session, self.cuda_provider)} "
            f"requested-provider={self.cuda_provider} execution_mode={self.execution_mode} "
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
                run_crops_batch_inpaint=lambda crops: self._inpaint_pytorch_crops_batch(
                    crops,
                    model_path=model_path,
                    repo_root=repo_root,
                    python_executable=python_executable,
                    max_side_px=DEFAULT_LAMA_PATCH_MAX_SIDE_PX,
                ),
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

    def _inpaint_pytorch_crops_batch(
        self,
        crops: list[tuple[np.ndarray, np.ndarray]],
        *,
        model_path: Path,
        repo_root: Path,
        python_executable: Path,
        max_side_px: int,
    ) -> list[np.ndarray]:
        if not crops:
            return []

        prepared: list[dict[str, Any]] = []
        for source_rgb, working_mask in crops:
            inference_mask = _expand_lama_inference_mask(
                working_mask,
                dilate_px=DEFAULT_LAMA_INFERENCE_MASK_DILATE_PX,
            )
            resized_rgb, resized_inference_mask, resized = _resize_lama_inputs(
                source_rgb,
                inference_mask,
                max_side_px=max_side_px,
            )
            prepared.append(
                {
                    "source_rgb": source_rgb,
                    "resized_rgb": resized_rgb,
                    "resized_inference_mask": resized_inference_mask,
                    "resized": resized,
                }
            )

        with tempfile.TemporaryDirectory(prefix="pdf2ppt-lama-batch-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            indir = temp_dir / "inputs"
            outdir = temp_dir / "outputs"
            indir.mkdir(parents=True, exist_ok=True)
            outdir.mkdir(parents=True, exist_ok=True)

            for index, item in enumerate(prepared):
                stem = f"patch{index:03d}"
                resized_page = Image.fromarray(item["resized_rgb"], mode="RGB")
                resized_mask_image = Image.fromarray(
                    (item["resized_inference_mask"] > 0).astype(np.uint8) * 255,
                    mode="L",
                )
                resized_page.save(indir / f"{stem}.png")
                resized_mask_image.save(indir / f"{stem}_mask001.png")

            logger.info(
                "LaMa PyTorch batch predict: crops=%s max_side=%s device=%s",
                len(prepared),
                max_side_px,
                self.device,
            )
            _run_lama_pytorch_prediction(
                python_executable=python_executable,
                repo_root=repo_root,
                model_path=model_path,
                device=self.device,
                indir=indir,
                outdir=outdir,
            )

            restored_list: list[np.ndarray] = []
            for index, item in enumerate(prepared):
                stem = f"patch{index:03d}"
                output_path = outdir / f"{stem}_mask001.png"
                if not output_path.exists():
                    output_candidates = sorted(outdir.glob(f"{stem}*.png"))
                    if len(output_candidates) == 1:
                        output_path = output_candidates[0]
                    else:
                        raise BackgroundInpaintingError(
                            f"Official LaMa batch prediction did not produce output for {stem} under {outdir}."
                        )
                restored_rgb = np.array(Image.open(output_path).convert("RGB"), dtype=np.uint8)
                if restored_rgb.shape[:2] != item["resized_rgb"].shape[:2]:
                    raise BackgroundInpaintingError(
                        f"Official LaMa batch output shape {restored_rgb.shape[:2]} "
                        f"did not match input shape {item['resized_rgb'].shape[:2]}."
                    )
                if item["resized"]:
                    source_rgb = item["source_rgb"]
                    restored_rgb = cv2.resize(
                        restored_rgb,
                        (source_rgb.shape[1], source_rgb.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                restored_list.append(restored_rgb)
        return restored_list


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

        logger.info(
            "Starting LaMa PyTorch worker (loading model to %s; first run may take 1-3 minutes)...",
            device,
        )
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
            "LaMa ONNX requires the onnxruntime package. Install pdf2ppt[cpu] (onnxruntime) for "
            "CPU inference, or pdf2ppt[gpu] (onnxruntime-gpu) for CUDA inference."
        ) from error


def _resolve_lama_execution_provider(requested_provider: str, available_providers: set[str]) -> str:
    """Resolve the ONNX Runtime execution provider to use.

    Phase 1.1: previously an unavailable ``requested_provider`` (e.g. a CUDA provider on a
    machine without a GPU) was a hard error. It now falls back to ``CPUExecutionProvider`` so
    the engine stays usable without a GPU. A hard error is still raised when neither the
    requested provider nor a CPU provider is available.
    """
    if requested_provider in available_providers:
        return requested_provider
    if "CPUExecutionProvider" in available_providers:
        return "CPUExecutionProvider"
    raise BackgroundInpaintingError(
        f"LaMa ONNX provider {requested_provider!r} is unavailable and no CPUExecutionProvider "
        f"fallback was found. Available providers: {sorted(available_providers)}"
    )


def _resolve_lama_session_provider(session: Any, requested_provider: str) -> str:
    try:
        providers = list(session.get_providers())
    except Exception:  # pragma: no cover - defensive against unusual ORT session objects
        return requested_provider
    return providers[0] if providers else requested_provider


def _get_lama_session(*, model_path: Path, cuda_provider: str, execution_mode: str) -> Any:
    cache_key = (str(model_path), cuda_provider, execution_mode)
    with _LAMA_SESSION_CACHE_LOCK:
        cached_session = _LAMA_SESSION_CACHE.get(cache_key)
    if cached_session is not None:
        return cached_session

    ort = _import_onnxruntime()
    available_providers = set(ort.get_available_providers())
    resolved_provider = _resolve_lama_execution_provider(cuda_provider, available_providers)
    if resolved_provider != cuda_provider:
        logger.info(
            "LaMa ONNX provider %r is unavailable; falling back to %r. Available providers: %s",
            cuda_provider,
            resolved_provider,
            sorted(available_providers),
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
            providers=[resolved_provider],
        )
    except Exception as error:
        raise BackgroundInpaintingError(f"Failed to initialize LaMa ONNX session: {error}") from error

    with _LAMA_SESSION_CACHE_LOCK:
        _LAMA_SESSION_CACHE.setdefault(cache_key, session)
        return _LAMA_SESSION_CACHE[cache_key]



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




__all__ = [
    "LamaOnnxCudaInpaintingEngine",
    "LamaPytorchInpaintingEngine",
    "_build_lama_model_inputs",
    "_build_lama_subprocess_env",
    "_coerce_lama_dimension",
    "_default_lama_python_candidates",
    "_fit_lama_inputs_to_model",
    "_get_lama_pytorch_worker",
    "_get_lama_session",
    "_import_onnxruntime",
    "_normalize_lama_output",
    "_pad_lama_inputs",
    "_read_lama_pytorch_response",
    "_resize_lama_inputs",
    "_resolve_lama_execution_provider",
    "_resolve_lama_fixed_input_size",
    "_resolve_lama_model_path",
    "_resolve_lama_python_executable",
    "_resolve_lama_pytorch_model_path",
    "_resolve_lama_repo_root",
    "_resolve_lama_session_provider",
    "_run_lama_pytorch_prediction",
    "_shutdown_lama_pytorch_workers",
    "_validate_lama_pytorch_runtime",
]
