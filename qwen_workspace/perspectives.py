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


PRESERVE = (
    "Use the reference images to depict the same subject. Preserve its identity, facial features, "
    "proportions, clothing or exterior design, colors, materials, pose, background, lighting and "
    "visual style. Do not add new features or redesign the subject. Produce one coherent image. "
)
PERSPECTIVES = {
    "three_quarter_body": Perspective(
        "3/4 body — mid-thigh up",
        "Make a close frontal UPPER-THIGH PORTRAIT of the same subject in the reference images. "
        "Magnify the subject and frame its upper body: complete head with a small margin above, "
        "shoulders, chest, abdomen, waist, hips, and only a short length of the upper thighs at "
        "the bottom. The bottom edge of the picture cuts across the TOP HALF of both thighs, "
        "just below the hips. Only the top portions of the thighs are inside the picture. "
        "For a subject with thigh armor or rectangular thigh vents in the reference, the lower "
        "edge passes through those vents, leaving only their upper portions visible. The thighs "
        "continue out of the photograph below the frame. Fill the square frame with this "
        "head-to-upper-thigh portrait. " + PRESERVE,
    ),
    "half_body": Perspective(
        "1/2 body — torso up",
        PRESERVE + "Create a half-body portrait showing the complete head, neck, shoulders, chest "
        "and upper abdomen. Camera at chest height, approximately 45 degrees toward the subject's "
        "front-left side, showing the front and left side of the torso with depth. Frame tightly "
        "from just above the top of the head to the middle of the abdomen. The lower image border "
        "cuts across the abdomen above the waist. The waist, pelvis, hips and legs are outside the "
        "frame. Crop the arms naturally at the bottom edge.",
    ),
    "full_body": Perspective(
        "Full body — head to feet",
        PRESERVE + "Move the camera approximately 45 degrees toward the subject's front-left "
        "side, at chest height. Show the front and left side with convincing depth. Keep the "
        "entire subject, from the complete head to both complete feet, inside the frame with "
        "a small surrounding margin. Preserve the original pose.",
    ),
    "birds_eye": Perspective(
        "Bird's-eye view",
        PRESERVE + "Place the camera high above and in front of the subject, looking downward "
        "approximately 60 degrees. Show the top of the head and shoulders with pronounced "
        "perspective foreshortening toward the feet. The subject keeps its original pose; the "
        "camera moves, rather than the subject bending or looking down. Keep the entire subject "
        "visible, from head to feet.",
    ),
    "worms_eye": Perspective(
        "Worm's-eye view",
        PRESERVE + "Place the camera close to the ground in front of the subject, looking upward "
        "toward its head. Show a pronounced low-angle perspective, with the feet closer to the "
        "camera and the head farther away. Keep the entire subject visible, including the "
        "complete head and both feet. Preserve the original pose.",
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
            references, true_cfg_scale, negative_prompt, reference_quality, vram_preset,
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
