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
    negative_prompt: str = ""
    true_cfg_scale: float = 1.0


# Passing updates: outputs/poses_expressions_cfg_20261017_a1/report.md.
# Other presets retain the historical defaults; see outputs/poses_six_robot_20261007/report.md.
POSES = {
    "wall_lean": Pose(
        "Wall lean",
        "Edit the subject in the reference image into the requested pose and expression. Preserve the "
        "same identity and underlying facial proportions: jaw and cheek planes, nose structure, eye color"
        " and iris scale. Allow the eyelids, brow region and mouth to move naturally for the requested "
        "expression. Preserve the original body proportions and silhouette, clothing or exterior "
        "boundaries, colors, surface materials, existing details, lighting and rendering style. Keep each"
        " region's original surface type during expression changes. Change pose, expression and framing "
        "as needed. Show a relaxed full-body stance against a plain wall. The shoulder or upper back "
        "visibly rests against the wall, with no gap at that contact. Place believable weight on one leg "
        "and slightly bend the other knee; let both arms hang comfortably. Give the subject a relaxed "
        "expression consistent with its identity. Frame the entire head, body and both complete feet with"
        " visible space above the head and below the feet. Add only the plain wall needed for contact and"
        " retain the reference setting style.",
        negative_prompt=(
            "Different identity, redesigned jaw or cheeks, changed nose structure, enlarged irises, changed "
            "eye color, enlarged lips, changed chest shape, narrowed waist, widened hips, altered limb "
            "proportions, invented clothing, exterior parts or accessories; replacing source surface "
            "materials, adding human skin or pores where absent in the reference, adding eyebrow hair where "
            "absent in the reference; extra or missing limbs, extra or fused fingers, broken wrists, "
            "disconnected joints, distorted hands, blurry details. Floating away from the wall, unsupported "
            "leaning, rigid symmetrical legs, tense raised arms, cropped head, cropped feet, feet touching "
            "the image border."
        ),
        true_cfg_scale=2.0,
    ),
    "victory_cheer": Pose(
        "Victory cheer",
        "Edit the subject in the reference image into the requested pose and expression. Preserve the "
        "same identity and underlying facial proportions: jaw and cheek planes, nose structure, eye color"
        " and iris scale. Allow the eyelids, brow region and mouth to move naturally for the requested "
        "expression. Preserve the original body proportions and silhouette, clothing or exterior "
        "boundaries, colors, surface materials, existing details, lighting and rendering style. Keep each"
        " region's original surface type during expression changes. Change pose, expression and framing "
        "as needed. Show unmistakable joyful celebration while looking toward the camera: both arms "
        "raised high above the shoulders, both hands visible, a clearly readable joyful closed-lip smile."
        " Curve the mouth corners upward and lift the cheeks naturally enough for unmistakable joy, "
        "keeping the original mouth width and narrow lip edges. Preserve the source nose length, nose tip"
        " and bridge structure, and underlying cheek and jaw planes during this smile. Keep the lip "
        "surface and lip-edge contrast consistent with the reference as the mouth corners rise. Use a "
        "natural balanced standing stance. Fit the entire head, raised hands, body and both complete feet"
        " inside the frame, with useful empty margins above the hands and below the feet. Recompose or "
        "move the viewpoint back enough to fit all of them. Retain the reference setting style.",
        negative_prompt=(
            "Different identity, redesigned jaw or cheeks, changed nose structure, enlarged irises, changed "
            "eye color, enlarged lips, changed chest shape, narrowed waist, widened hips, altered limb "
            "proportions, invented clothing, exterior parts or accessories; replacing source surface "
            "materials, adding human skin or pores where absent in the reference, adding eyebrow hair where "
            "absent in the reference; extra or missing limbs, extra or fused fingers, broken wrists, "
            "disconnected joints, distorted hands, blurry details. Lowered arms, hidden hands, neutral or sad"
            " mouth, unbalanced stance, cropped hands, cropped head, cropped feet, hands or feet touching the"
            " image border. Flattened or blunted nose relative to the reference, rounding away the original "
            "cheek or jaw planes, thickening the source lip edges, adding a thick dark lip outline absent "
            "from the source."
        ),
        true_cfg_scale=2.0,
    ),
    "flirty_smile": Pose(
        "Flirty smile",
        "Edit the subject in the reference image into the requested pose and expression. Preserve the "
        "same identity and underlying facial proportions: jaw and cheek planes, nose structure, eye color"
        " and iris scale. Allow the eyelids, brow region and mouth to move naturally for the requested "
        "expression. Preserve the original body proportions and silhouette, clothing or exterior "
        "boundaries, colors, surface materials, existing details, lighting and rendering style. Keep each"
        " region's original surface type during expression changes. Change pose, expression and framing "
        "as needed. Show a waist-up portrait looking directly at the camera, with a playful smile and "
        "subtle head tilt. Raise one hand so exactly one index fingertip lightly touches the lower lip. "
        "Make the fingertip-to-lower-lip contact clearly visible; gently curl the remaining fingers away "
        "from the mouth. The mouth visibly curves into a playful smile while retaining its original lip "
        "proportions. Show the entire head, both shoulders, the entire raised hand and torso down to the "
        "waist, with comfortable edge margins. Retain the reference setting style.",
        negative_prompt=(
            "Different identity, redesigned jaw or cheeks, changed nose structure, enlarged irises, changed "
            "eye color, enlarged lips, changed chest shape, narrowed waist, widened hips, altered limb "
            "proportions, invented clothing, exterior parts or accessories; replacing source surface "
            "materials, adding human skin or pores where absent in the reference, adding eyebrow hair where "
            "absent in the reference; extra or missing limbs, extra or fused fingers, broken wrists, "
            "disconnected joints, distorted hands, blurry details. Finger hovering away from the lip, finger "
            "inside the mouth, finger touching the nose or cheek, multiple fingers touching the mouth, hand "
            "hiding the smile, exaggerated pout, neutral mouth, cropped head, cropped raised hand, cropped "
            "waist."
        ),
        true_cfg_scale=2.0,
    ),
    "confident_hero": Pose(
        "Confident hero",
        "Edit the subject in the reference image into a confident full-body standing pose. Preserve the "
        "same identity and underlying facial geometry, including the source nose length and shape, cheek "
        "planes, jaw width and chin shape. Preserve original eye color and iris scale, body proportions "
        "and silhouette, clothing or exterior boundaries, colors, surface materials, details, lighting "
        "and rendering style. Allow only the small eyelid, brow and mouth movements needed for the "
        "expression while retaining the original facial surface and facial proportions. Both hands rest "
        "on the hips, elbows point outward, chest is lifted and shoulders draw back. Raise the chin only "
        "slightly, keeping both pupils aimed directly toward the camera. Plant both feet apart in a "
        "stable balanced stance. Give the subject a subtle proud smile with closed lips: lift the mouth "
        "corners only a little, preserving the original mouth width and lip thickness, with only shallow "
        "natural changes around the mouth corners. Keep the cheek planes and original jaw/chin silhouette"
        " recognizable during the small smile. Preserve original chest and waist shapes as posture "
        "changes. Frame the complete head, body and both complete feet with space above the head and "
        "below the feet. Retain the reference setting style.",
        negative_prompt=(
            "Different identity, redesigned jaw or cheeks, changed nose structure, enlarged irises, changed "
            "eye color, enlarged lips, changed chest shape, narrowed waist, widened hips, altered limb "
            "proportions, invented clothing, exterior parts or accessories; replacing source surface "
            "materials, adding human skin or pores where absent in the reference, adding eyebrow hair where "
            "absent in the reference; extra or missing limbs, extra or fused fingers, broken wrists, "
            "disconnected joints, distorted hands, blurry details. Hands away from hips, hidden hands, "
            "drooping elbows, slumped shoulders, neutral or frowning mouth, exaggerated chest or hip shape, "
            "cropped head, cropped feet, feet touching the image border."
        ),
        true_cfg_scale=2.0,
    ),
    "curious_thinker": Pose(
        "Curious thinker",
        "Edit the subject in the reference image into a waist-up thoughtful portrait. Preserve the same "
        "identity and the original underlying face geometry: original nose length and width, lip "
        "thickness, cheek and jaw planes, eye color and iris scale. Preserve original body proportions, "
        "silhouette, clothing or exterior boundaries, colors, surface materials, existing details, "
        "lighting and rendering style. Keep each region's original surface type. Change only the pose, "
        "expression and framing needed here. Turn the eyes toward the lens so both pupils look directly "
        "at the camera. Tilt the head only slightly. Make a small inquisitive change in one side of the "
        "existing brow region, with the forehead otherwise retaining its original smoothness and "
        "structure. Keep the original lips softly closed in a thoughtful expression. Raise one hand "
        "beneath the chin, palm facing upward. Lay its index finger sideways along the underside of the "
        "chin, making clear light contact under the chin; keep the index finger below the chin rather "
        "than extending up its side. Curl the remaining fingers gently below the chin, away from the "
        "cheek and mouth. Keep the fingertip contact and the separate curled fingers readable. Use a "
        "wider waist-up composition at lower magnification: move the viewpoint back enough to show the "
        "entire head, complete raised hand, torso down to the waist, and both complete shoulders "
        "including their outermost contours. Leave clearly visible background space outside each shoulder"
        " and the raised hand, above the head and below the waist. Fit the widest shoulder outline inside"
        " the frame with a useful lateral margin on both sides. Retain the reference setting style.",
        negative_prompt=(
            "Different identity, redesigned jaw or cheeks, changed nose structure, enlarged irises, changed "
            "eye color, enlarged lips, changed chest shape, narrowed waist, widened hips, altered limb "
            "proportions, invented clothing, exterior parts or accessories; replacing source surface "
            "materials, adding human skin or pores where absent in the reference, adding eyebrow hair where "
            "absent in the reference; extra or missing limbs, extra or fused fingers, broken wrists, "
            "disconnected joints, distorted hands, blurry details. Finger touching the lip or nose, finger "
            "hovering below the chin, hand covering the mouth, open mouth, broad smile, expressionless brow "
            "region, cropped head, cropped raised hand, cropped waist."
        ),
        true_cfg_scale=2.0,
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
    use_pose_settings: bool = False,
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
            references,
            pose.true_cfg_scale if use_pose_settings else true_cfg_scale,
            pose.negative_prompt if use_pose_settings else negative_prompt,
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
