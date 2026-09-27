"""Single lazy-loaded Diffusers pipeline for the local GPU."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

from .core import MODEL_ID, ORIGINAL_ASPECT, ROOT, Request, prepared_prompt

log = logging.getLogger(__name__)
_pipeline = None
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
MODEL_DIR = ROOT / "models" / "Qwen-Image-2.1"


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from diffusers import QwenImage21Pipeline
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run setup.ps1 and verify the NVIDIA driver.")
    complete = MODEL_DIR / ".download-complete"
    if not complete.exists() or complete.read_text(encoding="utf-8").strip() != MODEL_REVISION:
        log.info("Downloading/checking %s at %s (~33 GB on first run).", MODEL_ID, MODEL_DIR)
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR)
        complete.write_text(MODEL_REVISION, encoding="utf-8")
    log.info("Loading BF16 model with CPU offload. This may take several minutes.")
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    _pipeline = pipe
    log.info("Model ready.")
    return pipe


def infer(request: Request, timings: dict[str, float] | None = None) -> Image.Image:
    import torch

    started = time.perf_counter()
    pipe = get_pipeline()
    if timings is not None:
        timings["pipeline_load_seconds"] = time.perf_counter() - started
    images = []
    for path in request.references:
        with Image.open(path) as source:
            images.append(source.convert("RGBA" if source.mode == "RGBA" else "RGB"))
    model_width, model_height = request.width, request.height
    if request.aspect == ORIGINAL_ASPECT:
        model_width = (model_width + 15) // 16 * 16
        model_height = (model_height + 15) // 16 * 16
    kwargs = {
        "prompt": prepared_prompt(request),
        "width": model_width,
        "height": model_height,
        "num_inference_steps": request.steps,
        "generator": torch.Generator(device="cuda").manual_seed(request.seed),
        "true_cfg_scale": request.true_cfg_scale,
    }
    if request.true_cfg_scale > 1.0:
        # An empty string deliberately enables CFG against an unconditional prompt.
        kwargs["negative_prompt"] = request.negative_prompt
    if images:
        kwargs["image"] = images
    pipeline_start = time.perf_counter()
    if timings is not None:
        timings["input_prepare_seconds"] = pipeline_start - started - timings["pipeline_load_seconds"]

        def on_step_end(_pipe, step, _timestep, callback_kwargs):
            if step == 0:
                timings["first_step_seconds"] = time.perf_counter() - pipeline_start
            return callback_kwargs

        kwargs["callback_on_step_end"] = on_step_end
    with torch.inference_mode():
        result = pipe(**kwargs).images[0]
    pipeline_end = time.perf_counter()
    if timings is not None:
        timings["pipeline_seconds"] = pipeline_end - pipeline_start
    if request.aspect == ORIGINAL_ASPECT and result.size != (request.width, request.height):
        result = result.resize((request.width, request.height), Image.Resampling.LANCZOS)
    if timings is not None:
        timings["output_resize_seconds"] = time.perf_counter() - pipeline_end
    return result
