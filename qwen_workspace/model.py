"""Single lazy-loaded Diffusers pipeline for the local GPU."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .core import MODEL_ID, ROOT, Request, prepared_prompt

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


def infer(request: Request) -> Image.Image:
    import torch

    pipe = get_pipeline()
    images = []
    for path in request.references:
        with Image.open(path) as source:
            images.append(source.convert("RGBA" if source.mode == "RGBA" else "RGB"))
    kwargs = {
        "prompt": prepared_prompt(request),
        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.steps,
        "generator": torch.Generator(device="cuda").manual_seed(request.seed),
    }
    if images:
        kwargs["image"] = images
    with torch.inference_mode():
        return pipe(**kwargs).images[0]
