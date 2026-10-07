"""Single lazy-loaded Diffusers pipeline for the local GPU."""

from __future__ import annotations

import gc
import logging
import time
from types import MethodType
from pathlib import Path

from PIL import Image

from .core import (
    MODEL_ID, ORIGINAL_ASPECT, ROOT, VRAM_PRESETS, Request, prepared_prompt,
    reference_dimensions, validate_vram_preset,
)

log = logging.getLogger(__name__)
_pipeline = None
_pipeline_preset = None
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
MODEL_DIR = ROOT / "models" / "Qwen-Image-2.1"
INT8_TRANSFORMER_SKIP_MODULES = [
    "time_text_embed", "modulation", "norm_out", "proj_out", "img_in", "txt_in",
]


def _apply_int8_offload(layer, fn, recurse=True):
    """Move bnb's unregistered matmul state with recursive module.to()."""
    scales = layer.weight.SCB if layer.weight.SCB is not None else layer.state.SCB
    result = layer._vram_original_apply(fn, recurse=recurse)
    # Parent Module.to() bypasses Linear8bitLt.to(), leaving CB/SCB on CUDA.
    # Reattach them to the moved weight, ready for init_8bit_state next forward.
    layer.weight.CB = layer.weight.data
    layer.weight.SCB = fn(scales) if scales is not None else None
    layer.state.CB = layer.state.SCB = None
    return result


def _configure_int8_offload(component):
    from bitsandbytes.nn import Linear8bitLt

    for layer in component.modules():
        if isinstance(layer, Linear8bitLt):
            layer._vram_original_apply = layer._apply
            layer._apply = MethodType(_apply_int8_offload, layer)


def get_pipeline(vram_preset: str = "base"):
    global _pipeline, _pipeline_preset
    validate_vram_preset(vram_preset)
    if _pipeline is not None and _pipeline_preset == vram_preset:
        return _pipeline

    import torch
    from diffusers import QwenImage21Pipeline
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run setup.ps1 and verify the NVIDIA driver.")
    if _pipeline is not None:
        log.info("Unloading %s before switching to %s.", _pipeline_preset, vram_preset)
        old_pipe = _pipeline
        _pipeline = _pipeline_preset = None
        old_pipe.remove_all_hooks()
        del old_pipe
        gc.collect()
        torch.cuda.empty_cache()
    complete = MODEL_DIR / ".download-complete"
    if not complete.exists() or complete.read_text(encoding="utf-8").strip() != MODEL_REVISION:
        log.info("Downloading/checking %s at %s (~33 GB on first run).", MODEL_ID, MODEL_DIR)
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR)
        complete.write_text(MODEL_REVISION, encoding="utf-8")
    log.info("Loading %s with CPU offload. This may take several minutes.", VRAM_PRESETS[vram_preset])
    pipe = transformer = text_encoder = None
    try:
        components = {}
        if vram_preset != "base":
            import bitsandbytes  # Verify the optional backend before loading components.
            from diffusers import BitsAndBytesConfig, QwenImage21Transformer2DModel
            from transformers import BitsAndBytesConfig as TextBitsAndBytesConfig
            from transformers import Qwen3VLForConditionalGeneration

            config = ({"load_in_8bit": True} if vram_preset == "medium" else {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
            })
            transformer = QwenImage21Transformer2DModel.from_pretrained(
                MODEL_DIR, subfolder="transformer", dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(
                    **config,
                    **({"llm_int8_skip_modules": INT8_TRANSFORMER_SKIP_MODULES}
                       if vram_preset == "medium" else {}),
                ),
                local_files_only=True, device_map={"": "cuda:0"},
            )
            if vram_preset == "medium":
                _configure_int8_offload(transformer)
            transformer.to("cpu")
            torch.cuda.empty_cache()
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_DIR / "text_encoder", dtype=torch.bfloat16,
                quantization_config=TextBitsAndBytesConfig(**config),
                local_files_only=True, device_map={"": "cuda:0"},
            )
            if vram_preset == "medium":
                _configure_int8_offload(text_encoder)
            text_encoder.to("cpu")
            torch.cuda.empty_cache()
            components = {"transformer": transformer, "text_encoder": text_encoder}
        pipe = QwenImage21Pipeline.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, local_files_only=True, **components,
        )
        pipe.enable_model_cpu_offload()
    except Exception as exc:
        if pipe is not None:
            pipe.remove_all_hooks()
        # Clear local owners; the cache must never expose a partly loaded pipeline.
        pipe = transformer = text_encoder = None
        components = {}
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"Could not load {VRAM_PRESETS[vram_preset]}: {exc}. "
            "Rerun setup.cmd (or setup_runpod.sh) to install dependencies, "
            "close other GPU apps, or select Base and retry."
        ) from exc
    _pipeline = pipe
    _pipeline_preset = vram_preset
    log.info("Model ready.")
    return pipe


def infer(request: Request, timings: dict[str, float | int] | None = None) -> Image.Image:
    import torch

    started = time.perf_counter()
    cuda = getattr(torch, "cuda", None)
    track_vram_peaks = False
    if timings is not None and cuda is not None:
        try:
            if cuda.is_available():
                cuda.reset_peak_memory_stats()
                track_vram_peaks = True
        except (AttributeError, AssertionError, RuntimeError):
            pass
    pipe = get_pipeline(request.vram_preset)
    log.info("Starting %s inference with %s.", request.mode, VRAM_PRESETS[request.vram_preset])
    if timings is not None:
        timings["pipeline_load_seconds"] = time.perf_counter() - started
    images = []
    for index, path in enumerate(request.references, start=1):
        with Image.open(path) as source:
            image = source.convert("RGBA" if source.mode == "RGBA" else "RGB")
        original_size = image.size
        size = reference_dimensions(*image.size, request.reference_quality)
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        resized_note = (
            f" (resized from {original_size[0]}x{original_size[1]})"
            if image.size != original_size else ""
        )
        log.info(
            "Reference %d/%d resolution used by model: %dx%d%s",
            index, len(request.references), image.width, image.height, resized_note,
        )
        images.append(image)
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
        if track_vram_peaks:
            try:
                timings["peak_vram_allocated_bytes"] = cuda.max_memory_allocated()
                timings["peak_vram_reserved_bytes"] = cuda.max_memory_reserved()
            except (AttributeError, AssertionError, RuntimeError):
                pass
    return result
