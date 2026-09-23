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
QUALITY_PIXELS = {
    "Small (512×512)": 512**2,
    "Standard (~1 MP)": 1024**2,
    "2K (high memory)": 2048**2,
}
MAX_REFERENCES = 10
MAX_INPUT_PIXELS = 32_000_000


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


def dimensions(aspect: str, quality: str) -> tuple[int, int]:
    if aspect not in ASPECTS:
        raise ValueError("Choose a supported aspect ratio.")
    if quality not in QUALITY_PIXELS:
        raise ValueError("Choose Small, Standard, or 2K size.")
    x, y = ASPECTS[aspect]
    area = QUALITY_PIXELS[quality]
    factor = math.sqrt(area / (x * y))
    # The model's VAE needs dimensions divisible by 16.
    return max(256, round(x * factor / 16) * 16), max(256, round(y * factor / 16) * 16)


def validate_request(
    mode: str,
    prompt: str,
    aspect: str,
    quality: str,
    steps: int,
    seed: int | None,
    transparent: bool,
    references: list[str] | None = None,
) -> Request:
    if mode not in ("generate", "edit"):
        raise ValueError("Unknown mode.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Enter a prompt before starting.")
    if len(prompt) > 4000:
        raise ValueError("Keep the prompt under 4,000 characters.")
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
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.width * image.height > MAX_INPUT_PIXELS:
                    raise ValueError(f"{path.name} is too large; limit is 32 million pixels.")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not read image {path.name}.") from exc
    width, height = dimensions(aspect, quality)
    return Request(mode, prompt, aspect, quality, steps, seed, bool(transparent), paths, width, height)


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
        "mode": request.mode,
        "prompt": request.prompt,
        "effective_prompt": prepared_prompt(request),
        "aspect_ratio": request.aspect,
        "quality": request.quality,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "seed": request.seed,
        "transparent_requested": request.transparent,
        "image_mode": image.mode,
        "reference_images": [path.name for path in request.references],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, metadata_path
