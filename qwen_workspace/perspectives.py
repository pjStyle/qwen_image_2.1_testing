"""Reusable perspective prompts and sequential reference-image batches."""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import time
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .core import ORIGINAL_SIZE, OUTPUT_DIR, Request, save_result, validate_request
from .model import infer

log = logging.getLogger(__name__)
BATCHES_DIR = OUTPUT_DIR / "perspectives"


@dataclass(frozen=True)
class Perspective:
    label: str
    prompt: str
    negative_prompt: str = ""
    true_cfg_scale: float = 1.0


PRESERVE = (
    "Use the reference images to depict the same subject. Preserve its identity, facial features, "
    "proportions, clothing or exterior design, colors, materials, pose, background, lighting and "
    "visual style. Do not add new features or redesign the subject. Produce one coherent image. "
)
PERSPECTIVES = {
    'three_quarter_body': Perspective(
        '3/4 body — mid-thigh up',
        'Create a tightly framed head-to-mid-thigh portrait of the same subject shown in the reference '
        "image. Keep the reference camera direction and the subject's original body rotation, tilted "
        'stance, head tilt, gaze and arm positions. Keep the complete rounded crown fully inside the '
        'image. Leave a modest visible background margin above the complete crown, approximately 5 '
        'percent of the image height. Show the complete head, both shoulders, chest, abdomen, waist, hips'
        ' and the upper half of both thighs. The lower image border must pass directly through both thigh'
        ' surfaces at mid-thigh, halfway between hips and knees. Only the upper halves of the thighs are '
        'visible. The lower halves of both thighs, every part of both knees and all lower legs remain '
        'completely outside the photograph below its bottom border. The thigh surfaces continue beyond '
        'the bottom edge; their lower ends are not visible. Use a tight head-to-mid-thigh magnification: '
        'place the complete head near the upper edge with its visible margin and the midpoints of both '
        'thighs exactly at the lower edge. Preserve the reference identity, exact facial structure and '
        'expression, silhouette, body proportions, clothing or exterior design, existing surface details,'
        ' colors, materials, background, lighting and visual style. Preserve the original balance of '
        'smooth surfaces and exposed textured structures. Keep the reference eye size, eyelid shape, '
        'nose, mouth width, lip thickness, cheek planes and jaw shape. Render one coherent subject.',
        'full-length framing, visible knees, visible lower legs, visible feet, waist-only crop, cut-off '
        'head, changed pose, straightened stance, squared shoulders, changed body rotation, changed head '
        'tilt, altered facial identity, enlarged eyes, thickened lips, rounded cheeks, softened jaw, '
        'redesigned clothing or exterior, added surface markings, simplified exposed textures, changed '
        'colors, extra limbs, missing limbs, fused fingers, duplicated subject, blur, text, watermark, '
        'cropped crown, truncated crown, clipped top of head, head touching the top image edge, crown '
        'touching the top image edge, top border crossing the head, missing headroom above the crown',
        2.0,
    ),
    'half_body': Perspective(
        '1/2 body — torso up',
        'Create a close head-and-torso portrait of the same subject shown in the reference image. Keep '
        "the reference camera direction and the subject's original body rotation, tilted stance, head "
        'tilt, gaze and arm positions. Show the complete head, neck, both shoulders, chest and upper '
        'abdomen. The bottom image edge must cut across the middle of the abdomen, halfway between the '
        'chest and waist; the lower abdomen, waist, hips and legs lie outside the image. Crop the arms '
        'naturally where they cross the bottom image edge. Enlarge the subject to fill the frame with '
        'this crop. Preserve the reference identity, exact facial structure and expression, silhouette, '
        'body proportions, clothing or exterior design, existing surface details, colors, materials, '
        'background, lighting and visual style. Preserve the original balance of smooth surfaces and '
        'exposed textured structures. Keep the reference eye size, eyelid shape, nose, mouth width, lip '
        'thickness, cheek planes and jaw shape. Render one coherent subject.',
        'full-length framing, visible waist, visible hips, visible thighs, visible knees, visible feet, '
        'chest-only crop, cut-off head, changed pose, straightened stance, squared shoulders, changed '
        'body rotation, changed head tilt, altered facial identity, enlarged eyes, thickened lips, '
        'rounded cheeks, softened jaw, redesigned clothing or exterior, added surface markings, '
        'simplified exposed textures, changed colors, extra limbs, missing limbs, duplicated subject, '
        'blur, text, watermark',
        2.0,
    ),
    'full_body': Perspective(
        'Full body — head to feet',
        'Create a complete full-body view of the same subject shown in the reference image. Keep the '
        "reference camera direction, camera height and perspective. Preserve the subject's original body "
        'rotation, tilted stance, weight distribution, head tilt, gaze, arm positions and foot placement.'
        ' Show the entire subject from the complete head to both complete feet, with a small margin above'
        ' the head and below the feet. Preserve the reference identity, exact facial structure and '
        'expression, silhouette, body proportions, clothing or exterior design, existing surface details,'
        ' colors, materials, background, lighting and visual style. Preserve the original balance of '
        'smooth surfaces and exposed textured structures. Keep the reference eye size, eyelid shape, '
        'nose, mouth width, lip thickness, cheek planes and jaw shape. Render one coherent subject.',
        'cropped head, cropped feet, cropped toes, close portrait, changed pose, straightened stance, '
        'squared shoulders, changed body rotation, changed head tilt, changed foot placement, altered '
        'facial identity, enlarged eyes, thickened lips, rounded cheeks, softened jaw, redesigned '
        'clothing or exterior, added surface markings, simplified exposed textures, changed colors, extra'
        ' limbs, missing limbs, fused fingers, duplicated subject, blur, text, watermark',
        2.0,
    ),
    'birds_eye': Perspective(
        "Bird's-eye view",
        "Create a whole-body bird's-eye view of the same subject shown in the reference image. Move the "
        'camera high above the subject, on the same side of the subject as the reference camera, and aim '
        'it downward approximately 60 degrees below horizontal toward the subject. This is a strong '
        'overhead perspective: the top of the head and upper surfaces of the shoulders are visible, and '
        'the body recedes and foreshortens toward the feet. Fit the complete head and both complete feet '
        'inside the frame with small margins. The subject keeps its original body rotation, tilted '
        'stance, weight distribution, head tilt, arm positions and foot placement; only the camera '
        'position changes. Preserve the reference identity, exact facial structure and expression, '
        'silhouette, body proportions, clothing or exterior design, existing surface details, colors, '
        'materials, background, lighting and visual style as they appear from this elevated viewpoint. '
        'Preserve the original balance of smooth surfaces and exposed textured structures. Render one '
        'coherent subject.',
        'eye-level camera, weak downward angle, low-angle camera, flat front elevation, subject bending '
        'forward, subject bowing head, changed pose, straightened stance, squared shoulders, changed body'
        ' rotation, cropped head, cropped feet, altered facial identity, redesigned clothing or exterior,'
        ' added surface markings, simplified exposed textures, changed colors, stretched torso, extra '
        'limbs, missing limbs, disconnected joints, duplicated subject, blur, text, watermark',
        3.0,
    ),
    'worms_eye': Perspective(
        "Worm's-eye view",
        "Create a whole-body worm's-eye view of the same subject shown in the reference image. Place the "
        "camera at floor level near the subject's feet, on the same side of the subject as the reference "
        'camera, and aim it steeply upward toward the head. Use pronounced upward perspective: the feet '
        'and lower legs are closest to the camera and appear larger, the torso recedes upward, and the '
        'head appears farther away. Show the undersides of projecting surfaces where this low viewpoint '
        'makes them visible. Use a wider composition and center the entire projected silhouette, '
        'including the feet, within the image. Keep the complete head and both complete feet, including '
        'every toe, fully inside the frame. Leave generous empty background margins on both sides of the '
        'feet, at least 10 percent of the image width beyond the outermost toe on each side, with clear '
        'margin below the soles and above the head. Keep the floor-level camera and steep upward aim '
        'while widening the field of view enough to fit these margins. The subject keeps its original '
        'body rotation, tilted stance, weight distribution, head tilt, arm positions and foot placement; '
        'only the camera position changes. Preserve the reference identity, exact facial structure and '
        'expression, silhouette, body proportions, clothing or exterior design, existing surface details,'
        ' colors, materials, background, lighting and visual style as they appear from this low '
        'viewpoint. Preserve the original balance of smooth surfaces and exposed textured structures. '
        'Render one coherent subject.',
        'eye-level camera, chest-height camera, weak upward angle, overhead camera, flat front elevation,'
        ' subject leaning toward camera, changed pose, straightened stance, squared shoulders, changed '
        'body rotation, cropped head, cropped feet, altered facial identity, redesigned clothing or '
        'exterior, added surface markings, simplified exposed textures, changed colors, stretched torso, '
        'extra limbs, missing limbs, disconnected joints, duplicated subject, blur, text, watermark',
        2.0,
    ),
    "face_closeup": Perspective(
        "Extreme face close-up",
        PRESERVE + "Create an extreme face close-up directly from the front at eye level. Zoom "
        "in until the face alone fills virtually the entire frame. The top boundary cuts the "
        "upper forehead and the bottom boundary cuts the lower chin. Show both eyes, nose and "
        "mouth with narrow side margins. Exclude the neck, shoulders, chest and body. Preserve "
        "the exact facial geometry, expression, eye color and existing surface details from "
        "the reference. Reveal natural fine texture without inventing new facial features, "
        "panels, markings or accessories.",
    ),
}


def prepare_requests(
    references: list[str] | None,
    selected: list[str] | None,
    additional_instructions: str = "",
    prompt_overrides: dict[str, str] | None = None,
    *,
    aspect: str = "Original image",
    quality: str = ORIGINAL_SIZE,
    steps: int = 40,
    seed: int = 42,
    transparent: bool = False,
    true_cfg_scale: float = 1.0,
    negative_prompt: str = "",
    reference_quality: str = ORIGINAL_SIZE,
    vram_preset: str = "medium",
    use_perspective_settings: bool = False,
) -> list[tuple[str, Request]]:
    """Validate every selected view before starting; share one seed and original references."""
    if not selected:
        raise ValueError("Choose at least one perspective.")
    if any(key not in PERSPECTIVES for key in selected):
        raise ValueError("Choose a supported perspective.")
    overrides = prompt_overrides or {}
    requests = []
    batch_seed = seed
    for key, perspective in PERSPECTIVES.items():
        if key not in selected:
            continue
        prompt = overrides.get(key, perspective.prompt).strip()
        if not prompt:
            raise ValueError(f"Enter a prompt for {perspective.label}.")
        if additional_instructions.strip():
            prompt += "\n\nAdditional instructions: " + additional_instructions.strip()
        request = validate_request(
            "edit", prompt, aspect, quality, steps, batch_seed, transparent,
            references,
            perspective.true_cfg_scale if use_perspective_settings else true_cfg_scale,
            perspective.negative_prompt if use_perspective_settings else negative_prompt,
            reference_quality, vram_preset,
        )
        batch_seed = request.seed
        requests.append((key, request))
    return requests


@dataclass
class Batch:
    folder: Path
    requests: list[tuple[str, Request]]
    manifest: dict

    def save(self) -> None:
        (self.folder / "batch.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )


def create_batch(requests: list[tuple[str, Request]], output_root: Path = BATCHES_DIR) -> Batch:
    if not requests:
        raise ValueError("Choose at least one perspective.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = output_root / f"{stamp}_{secrets.token_hex(4)}"
    references_dir = folder / "references"
    references_dir.mkdir(parents=True)
    copied_references = []
    for index, reference in enumerate(requests[0][1].references, 1):
        target = references_dir / f"reference_{index:02d}{reference.suffix.lower()}"
        shutil.copy2(reference, target)
        copied_references.append(target)
    batch = Batch(
        folder,
        [(key, replace(request, references=tuple(copied_references))) for key, request in requests],
        {
            "id": folder.name,
            "status": "ready",
            "seed": requests[0][1].seed,
            "selected": [key for key, _ in requests],
            "reference_filenames": [p.name for p in requests[0][1].references],
            "results": [],
        },
    )
    batch.save()
    return batch


def generate_batch(batch: Batch) -> Iterator[dict]:
    """Persist each result before yielding; retain completed images if a later view fails."""
    batch.manifest["status"] = "running"
    batch.save()
    try:
        for key, request in batch.requests:
            batch.manifest["current"] = key
            batch.save()
            started = time.perf_counter()
            timings = {}
            image = infer(request, timings)
            image_path, metadata_path = save_result(image, request, batch.folder / key)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({
                "perspective": key,
                "perspective_label": PERSPECTIVES[key].label,
                "batch_id": batch.folder.name,
                "timings": timings,
                "total_seconds": time.perf_counter() - started,
            })
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            result = {
                "perspective": key,
                "label": PERSPECTIVES[key].label,
                "image": str(image_path.relative_to(batch.folder)),
                "metadata": str(metadata_path.relative_to(batch.folder)),
                "total_seconds": metadata["total_seconds"],
            }
            batch.manifest["results"].append(result)
            batch.save()
            yield result
        batch.manifest["status"] = "completed"
        batch.manifest.pop("current", None)
    except Exception as exc:
        batch.manifest["status"] = "failed"
        batch.manifest["error"] = str(exc)
        log.exception("Perspective batch %s failed", batch.folder.name)
        raise
    finally:
        if batch.manifest["status"] == "running":
            batch.manifest["status"] = "stopped"
        batch.save()


def zip_batch(batch: Batch) -> Path:
    """Include PNGs, prompts/settings, manifest and original uploaded references."""
    archive = batch.folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(batch.folder.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(batch.folder))
    return archive
