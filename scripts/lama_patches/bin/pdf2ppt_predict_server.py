#!/usr/bin/env python3
"""Long-lived LaMa predictor for pdf2ppt. Loads the checkpoint once and serves JSON-line jobs on stdin."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger(__name__)


def _resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower() or "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        normalized = "cpu"
    return torch.device(normalized)


def _load_model(*, model_path: Path, device: torch.device) -> tuple[object, OmegaConf]:
    train_config_path = model_path / "config.yaml"
    with train_config_path.open("r", encoding="utf-8") as config_file:
        train_config = OmegaConf.create(yaml.safe_load(config_file))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"
    checkpoint_path = model_path / "models" / "best.ckpt"
    map_location = device if device.type == "cuda" else "cpu"
    model = load_checkpoint(train_config, str(checkpoint_path), strict=False, map_location=map_location)
    model.freeze()
    if not train_config.get("refine", False):
        model.to(device)
    return model, train_config


def _predict_indir(
    *,
    model: object,
    train_config: OmegaConf,
    device: torch.device,
    indir: Path,
    outdir: Path,
    img_suffix: str,
) -> None:
    indir_text = str(indir)
    if not indir_text.endswith("/"):
        indir_text += "/"
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = make_default_val_dataset(indir_text, kind="default", img_suffix=img_suffix, pad_out_to_modulo=8)
    out_key = "inpainted"
    out_ext = ".png"
    for image_index in range(len(dataset)):
        mask_filename = dataset.mask_filenames[image_index]
        output_path = outdir / (
            Path(mask_filename[len(indir_text) :]).with_suffix(out_ext).name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        batch = default_collate([dataset[image_index]])
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch["mask"] = (batch["mask"] > 0) * 1
            batch = model(batch)
            result = batch[out_key][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get("unpad_to_size", None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                result = result[:orig_height, :orig_width]
        result = np.clip(result * 255, 0, 255).astype("uint8")
        cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def _write_response(*, ok: bool, message: str = "") -> None:
    payload = {"ok": ok, "message": message}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="Persistent LaMa prediction server for pdf2ppt.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-suffix", default=".png")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model, train_config = _load_model(model_path=args.model_path.resolve(), device=device)
    _write_response(ok=True, message=f"ready device={device}")

    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "shutdown":
            _write_response(ok=True, message="shutdown")
            return 0
        try:
            job = json.loads(stripped)
            indir = Path(str(job["indir"])).resolve()
            outdir = Path(str(job["outdir"])).resolve()
            _predict_indir(
                model=model,
                train_config=train_config,
                device=device,
                indir=indir,
                outdir=outdir,
                img_suffix=str(job.get("img_suffix", args.img_suffix)),
            )
            _write_response(ok=True)
        except Exception as error:
            LOGGER.exception("Prediction job failed")
            _write_response(ok=False, message=f"{error}\n{traceback.format_exc()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
