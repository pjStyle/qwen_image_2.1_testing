"""Validation, model access, and local output handling."""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


MODEL_ID = "Qwen/Qwen-Image-2.1"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
}
ORIGINAL_ASPECT = "Original image"
ORIGINAL_SIZE = "Preserve input resolution"
QUALITY_PIXELS = {
    "Small (512×512)": 512**2,
    "Standard (~1 MP)": 1024**2,
    "1.5 MP": 1_500_000,
    "2K (high memory)": 2048**2,
}
REFERENCE_QUALITY_PIXELS = {"Extra small (256×256)": 256**2, **QUALITY_PIXELS}
MAX_REFERENCES = 10
MAX_INPUT_PIXELS = 32_000_000
VRAM_PRESETS = {
    "base": "Base — BF16",
    "medium": "Medium VRAM — 8-bit",
    "low": "Low VRAM — 4-bit",
}


def validate_vram_preset(value: str) -> str:
    if not isinstance(value, str) or value not in VRAM_PRESETS:
        raise ValueError("Choose a supported VRAM preset.")
    return value


@dataclass(frozen=True)
class Request:
    mode: str
    prompt: str
    aspect: str
    quality: str
    steps: int
    seed: int
    transparent: bool
    references: tuple[Path, ...]
    width: int
    height: int
    true_cfg_scale: float = 1.0
    negative_prompt: str = ""
    reference_quality: str = ORIGINAL_SIZE
    vram_preset: str = "base"


def dimensions(aspect: str, quality: str) -> tuple[int, int]:
    if aspect not in ASPECTS:
        raise ValueError("Choose a supported aspect ratio.")
    return dimensions_for_ratio(*ASPECTS[aspect], quality)


def dimensions_for_ratio(x: int, y: int, quality: str) -> tuple[int, int]:
    if quality not in QUALITY_PIXELS:
        raise ValueError("Choose Small, Standard, 1.5 MP, or 2K size.")
    area = QUALITY_PIXELS[quality]
    factor = math.sqrt(area / (x * y))
    # The model's VAE needs dimensions divisible by 16.
    return max(256, round(x * factor / 16) * 16), max(256, round(y * factor / 16) * 16)


def original_dimensions(x: int, y: int, quality: str) -> tuple[int, int]:
    if quality not in QUALITY_PIXELS:
        raise ValueError("Choose Small, Standard, 1.5 MP, or 2K size.")
    factor = max(math.sqrt(QUALITY_PIXELS[quality] / (x * y)), 256 / min(x, y))
    width = round(x * factor)
    height = round(width * y / x)
    return width, height


def reference_dimensions(width: int, height: int, quality: str) -> tuple[int, int]:
    """Return source dimensions, or proportionally downsize to a pixel-area cap."""
    if quality == ORIGINAL_SIZE:
        return width, height
    if quality not in REFERENCE_QUALITY_PIXELS:
        raise ValueError("Choose a supported reference image size.")
    max_pixels = REFERENCE_QUALITY_PIXELS[quality]
    if width * height <= max_pixels:
        return width, height
    factor = math.sqrt(max_pixels / (width * height))
    resized_width = max(1, int(width * factor))
    resized_height = max(1, int(height * factor))
    if resized_width * resized_height > max_pixels:
        if width >= height:
            resized_width = max(1, max_pixels // resized_height)
        else:
            resized_height = max(1, max_pixels // resized_width)
    return resized_width, resized_height


def validate_request(
    mode: str,
    prompt: str,
    aspect: str,
    quality: str,
    steps: int,
    seed: int | None,
    transparent: bool,
    references: list[str] | None = None,
    true_cfg_scale: float = 1.0,
    negative_prompt: str = "",
    reference_quality: str = ORIGINAL_SIZE,
    vram_preset: str = "base",
) -> Request:
    validate_vram_preset(vram_preset)
    if mode not in ("generate", "edit"):
        raise ValueError("Unknown mode.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Enter a prompt before starting.")
    if len(prompt) > 4000:
        raise ValueError("Keep the prompt under 4,000 characters.")
    if isinstance(true_cfg_scale, bool) or not isinstance(true_cfg_scale, (int, float)):
        raise ValueError("CFG scale must be a number from 1.0 to 10.0.")
    true_cfg_scale = float(true_cfg_scale)
    if not math.isfinite(true_cfg_scale) or not 1.0 <= true_cfg_scale <= 10.0:
        raise ValueError("CFG scale must be from 1.0 to 10.0.")
    if negative_prompt is None:
        negative_prompt = ""
    if not isinstance(negative_prompt, str):
        raise ValueError("Negative prompt must be text.")
    negative_prompt = negative_prompt.strip()
    if len(negative_prompt) > 4000:
        raise ValueError("Keep the negative prompt under 4,000 characters.")
    if reference_quality != ORIGINAL_SIZE and reference_quality not in REFERENCE_QUALITY_PIXELS:
        raise ValueError("Choose a supported reference image size.")
    if not isinstance(steps, int) or not 1 <= steps <= 80:
        raise ValueError("Steps must be between 1 and 80.")
    if seed is None or seed == -1:
        seed = secrets.randbelow(2**32)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be -1 (random) or from 0 to 4,294,967,295.")
    paths = tuple(Path(path) for path in (references or []))
    if mode == "generate" and paths:
        raise ValueError("Reference images belong in Edit.")
    if mode == "edit" and not 1 <= len(paths) <= MAX_REFERENCES:
        raise ValueError("Upload 1 to 10 reference images for Edit.")
    original_size = None
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.width * image.height > MAX_INPUT_PIXELS:
                    raise ValueError(f"{path.name} is too large; limit is 32 million pixels.")
                if original_size is None:
                    original_size = image.size
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not read image {path.name}.") from exc
    if aspect == ORIGINAL_ASPECT:
        if mode != "edit" or original_size is None:
            raise ValueError("Original image aspect ratio requires an Edit reference image.")
        width, height = original_size if quality == ORIGINAL_SIZE else original_dimensions(*original_size, quality)
    else:
        if quality == ORIGINAL_SIZE:
            raise ValueError("Preserve input resolution requires the Original image aspect ratio in Edit.")
        width, height = dimensions(aspect, quality)
    return Request(
        mode, prompt, aspect, quality, steps, seed, bool(transparent), paths, width, height,
        true_cfg_scale, negative_prompt, reference_quality, vram_preset,
    )


def prepared_prompt(request: Request) -> str:
    if not request.transparent:
        return request.prompt
    return f"This is an RGBA image with transparency. {request.prompt} The image has alpha channel and the background is transparent."


def save_result(image: Image.Image, request: Request, output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{stamp}_{request.mode}_{request.seed}"
    image_path = output_dir / f"{name}.png"
    metadata_path = output_dir / f"{name}.json"
    image.save(image_path, format="PNG")
    metadata: dict[str, Any] = {
        "model": MODEL_ID,
        "vram_preset": request.vram_preset,
        "mode": request.mode,
        "prompt": request.prompt,
        "effective_prompt": prepared_prompt(request),
        "aspect_ratio": request.aspect,
        "quality": request.quality,
        "reference_quality": request.reference_quality,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "seed": request.seed,
        "true_cfg_scale": request.true_cfg_scale,
        "negative_prompt": request.negative_prompt,
        "transparent_requested": request.transparent,
        "image_mode": image.mode,
        "reference_images": [path.name for path in request.references],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, metadata_path
