"""Reusable pose and expression prompts and sequential reference-image batches."""

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
BATCHES_DIR = OUTPUT_DIR / "poses"


@dataclass(frozen=True)
class Pose:
    label: str
    prompt: str


# Reviewed defaults; see outputs/poses_six_robot_20261007/report.md.
POSES = {
    "wall_lean": Pose(
        "Wall lean",
        "Edit the subject in the reference images. Preserve the same subject identity, facial structure, "
        "clothing or exterior design, colors, materials, fine details, lighting, and visual style. Keep "
        "the subject recognizable. Change only the body pose, facial expression, and composition needed "
        "for the requested action. Show the subject in a relaxed full-body stance leaning against a plain "
        "wall, with a shoulder or the upper back visibly resting against the wall. Weight rests naturally "
        "on one leg, the other knee is slightly bent, and the arms hang comfortably. Keep the complete "
        "body and both feet in frame. Add only the wall needed for this pose; retain the reference "
        "background style.",
    ),
    "victory_cheer": Pose(
        "Victory cheer",
        "Edit the subject in the reference images. Preserve the same subject identity, facial structure, "
        "clothing or exterior design, colors, materials, fine details, lighting, and visual style. Keep "
        "the subject recognizable. Change only the body pose, facial expression, and composition needed "
        "for the requested action. Show the subject celebrating a victory, with both arms raised high "
        "above the shoulders and a joyful smile directed toward the camera. Make the uplifted arms and "
        "triumphant expression unmistakable. Show the complete body, both hands, and both feet within the "
        "frame, with natural balanced anatomy.",
    ),
    "flirty_smile": Pose(
        "Flirty smile",
        "Edit the subject in the reference images. Preserve the same subject identity, facial structure, "
        "clothing or exterior design, colors, materials, fine details, lighting, and visual style. Keep "
        "the subject recognizable. Change only the body pose, facial expression, and composition needed "
        "for the requested action. Show a waist-up portrait of the subject looking directly at the camera "
        "with a playful, flirty smile. One hand is raised near the mouth: exactly one index finger "
        "lightly touches the lower lip, while the remaining fingers curl naturally away from the mouth. "
        "The lips curve into a visible smile and the head has a subtle inviting tilt. Keep the entire "
        "head, shoulders, raised hand, and torso down to the waist in frame.",
    ),
    "confident_hero": Pose(
        "Confident hero",
        "Edit the subject in the reference images. Preserve the same subject identity, facial structure, "
        "clothing or exterior design, colors, materials, fine details, lighting, and visual style. Keep "
        "the subject recognizable. Change only the body pose, facial expression, and composition needed "
        "for the requested action. Show the subject standing in a confident heroic pose, with both hands "
        "resting firmly on the hips, elbows pointing outward, chest lifted, shoulders drawn back, and "
        "chin slightly raised. The feet are planted apart in a stable stance. Give the subject a subtle "
        "proud smile directed toward the camera. Show the complete body and both feet inside the frame. "
        "Keep the reference background style.",
    ),
    "curious_thinker": Pose(
        "Curious thinker",
        "Edit the subject in the reference images. Preserve the same subject identity, facial structure, "
        "clothing or exterior design, colors, materials, fine details, lighting, and visual style. Keep "
        "the subject recognizable. Change only the body pose, facial expression, and composition needed "
        "for the requested action. Show a waist-up portrait of the subject looking at the camera with an "
        "inquisitive expression. One hand rests naturally under the chin in a thinking gesture, with the "
        "index finger along the underside of the chin and the other fingers curled gently below it. Tilt "
        "the head slightly to one side, lift one eyebrow, and keep the lips softly closed in a thoughtful "
        "expression. Keep the entire head, shoulders, raised hand, and torso down to the waist visible. "
        "Keep the reference background style.",
    ),
    "surprised_reaction": Pose(
        "Surprised reaction",
        "Edit the subject in the reference images. Preserve the same subject identity, facial structure, "
        "clothing or exterior design, colors, materials, fine details, lighting, and visual style. Keep "
        "the subject recognizable. Change only the body pose, facial expression, and composition needed "
        "for the requested action. Show a waist-up portrait of the subject reacting with restrained, "
        "realistic surprise while looking directly at the camera. Preserve the original eye color, iris "
        "size, eye shape, nose, facial proportions, and facial surface. Widen the eyelids slightly and "
        "lift the eyebrows to express surprise, without enlarging or replacing the eyes. Part the lips "
        "into a small open mouth. Both hands are lifted naturally near the cheeks, one on each side of "
        "the face, with open palms and fingers pointing upward. Keep the palms close beside the cheeks "
        "without covering the face. Show the complete head, both hands, shoulders, and torso down to the "
        "waist. Keep the reference background style.",
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
    vram_preset: str = "base",
) -> list[tuple[str, Request]]:
    """Validate every selected pose before starting; share one seed and original references."""
    if not selected:
        raise ValueError("Choose at least one pose or expression.")
    if any(key not in POSES for key in selected):
        raise ValueError("Choose a supported pose or expression.")
    overrides = prompt_overrides or {}
    requests = []
    batch_seed = seed
    for key, pose in POSES.items():
        if key not in selected:
            continue
        prompt = overrides.get(key, pose.prompt).strip()
        if not prompt:
            raise ValueError(f"Enter a prompt for {pose.label}.")
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
        raise ValueError("Choose at least one pose or expression.")
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
    """Persist each result before yielding; retain completed images if a later pose fails."""
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
                "pose": key,
                "pose_label": POSES[key].label,
                "batch_id": batch.folder.name,
                "timings": timings,
                "total_seconds": time.perf_counter() - started,
            })
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            result = {
                "pose": key,
                "label": POSES[key].label,
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
        log.exception("Pose batch %s failed", batch.folder.name)
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
