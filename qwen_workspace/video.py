"""Extract, edit or upscale, resume, and encode local video jobs."""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
from PIL import Image, UnidentifiedImageError

from .core import (ASPECTS, ORIGINAL_ASPECT, ORIGINAL_SIZE, OUTPUT_DIR,
                   Request, dimensions, original_dimensions, validate_vram_preset)
from .model import infer


JOBS_DIR = OUTPUT_DIR / "video_jobs"
log = logging.getLogger(__name__)
DEFAULT_PROMPT = (
    "Upscale and refine this video frame. Preserve the exact subjects, identities, "
    "composition, camera angle, colors, lighting, and original style. Add only natural detail; "
    "do not add, remove, or move objects."
)
EDIT_PROMPT_HINT = "Describe one change to make in every frame, for example: Change the red car to blue."
Progress = Callable[[int, int, float | None], None]


class PauseRequested(Exception):
    """Raised after a completed frame when the user requests a pause."""


def _ffmpeg(*args: str) -> None:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1500:]
        raise RuntimeError(f"FFmpeg failed: {detail or 'unknown error'}")


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}", job_id or ""):
        raise ValueError("Invalid job ID.")
    return JOBS_DIR / job_id


def _manifest_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def load_job(job_id: str) -> dict:
    path = _manifest_path(job_id)
    if not path.is_file():
        raise ValueError("Video job not found. Check the job ID in the outputs/video_jobs folder.")
    return json.loads(path.read_text(encoding="utf-8"))


def job_mode(job: dict) -> str:
    # Jobs saved before Video Edit was added have no mode field.
    return job.get("mode", "upscale")


def require_job_mode(job_id: str, mode: str) -> dict:
    job = load_job(job_id)
    if job_mode(job) != mode:
        raise ValueError(f"This job belongs to the Video {'Edit' if job_mode(job) == 'edit' else 'Upscale'} tab.")
    return job


def _processed_name(job: dict) -> str:
    return "edited" if job_mode(job) == "edit" else "upscaled"


def _write_job(job: dict) -> None:
    path = _manifest_path(job["id"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _validate_extraction(fps: int, start: float, end: float | None) -> None:
    if not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("Output FPS must be an integer from 1 to 60.")
    if start < 0:
        raise ValueError("Start time must be zero or greater.")
    if end is not None and end <= start:
        raise ValueError("End time must be later than start time.")


def numbered_frames(directory: Path) -> list[Path]:
    frames = [p for p in directory.glob("*.png") if p.stem.isdigit()]
    return sorted(frames, key=lambda p: int(p.stem))


def target_dimensions(width: int, height: int, long_edge: int) -> tuple[int, int]:
    if not isinstance(long_edge, int) or not 256 <= long_edge <= 2048 or long_edge % 16:
        raise ValueError("Target long edge must be a multiple of 16 from 256 to 2048.")
    if width < 1 or height < 1:
        raise ValueError("Invalid source frame dimensions.")
    if width >= height:
        return long_edge, max(16, round(height * long_edge / width / 16) * 16)
    return max(16, round(width * long_edge / height / 16) * 16), long_edge


def model_dimensions(width: int, height: int) -> tuple[int, int]:
    """Round up to the 32-pixel grid required by Qwen Image 2.1."""
    return ((width + 31) // 32 * 32, (height + 31) // 32 * 32)


def create_job(video_path: str, fps: int, start: float = 0, end: float | None = None, mode: str = "upscale") -> dict:
    _validate_extraction(fps, start, end)
    if mode not in ("upscale", "edit"):
        raise ValueError("Unknown video mode.")
    source = Path(video_path)
    if not source.is_file():
        raise ValueError("Upload a video before extracting frames.")
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(4)
    directory = _job_dir(job_id)
    frame_dir = directory / "source_frames"
    frame_dir.mkdir(parents=True)
    copied_source = directory / ("source" + source.suffix.lower())
    shutil.copy2(source, copied_source)
    job = {
        "id": job_id,
        "mode": mode,
        "source_name": source.name,
        "source_file": copied_source.name,
        "fps": fps,
        "start_seconds": start,
        "end_seconds": end,
        "status": "extracting",
        "settings": None,
        "completed_frames": 0,
    }
    _write_job(job)
    try:
        trim = ["-ss", str(start)] if start else []
        trim += ["-t", str(end - start)] if end is not None else []
        _ffmpeg(
            *trim, "-i", str(copied_source), "-map", "0:v:0", "-an", "-vf", f"fps={fps}",
            "-start_number", "0", str(frame_dir / "%06d.png"),
        )
        frames = numbered_frames(frame_dir)
        if not frames:
            raise ValueError("No frames were extracted. Check the video and time range.")
        with Image.open(frames[0]) as first:
            job["source_width"], job["source_height"] = first.size
        job["frame_count"] = len(frames)
        job["output_duration_seconds"] = len(frames) / fps
        job["status"] = "extracted"
        _write_job(job)
        return job
    except Exception as exc:
        job["status"] = "extraction_failed"
        job["error"] = str(exc)
        _write_job(job)
        raise


def _valid_frame(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return image.format == "PNG" and image.size == expected_size and image.verify() is None
    except (UnidentifiedImageError, OSError):
        return False


def _settings(long_edge: int, steps: int, seed: int, prompt: str, source_size: tuple[int, int]) -> dict:
    width, height = target_dimensions(*source_size, long_edge)
    if not isinstance(steps, int) or not 1 <= steps <= 80:
        raise ValueError("Steps must be from 1 to 80.")
    prompt = (prompt or "").strip()
    if not prompt or len(prompt) > 4000:
        raise ValueError("Enter a prompt of up to 4,000 characters.")
    if seed == -1:
        seed = secrets.randbelow(2**32)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be -1 or from 0 to 4,294,967,295.")
    return {"long_edge": long_edge, "width": width, "height": height, "steps": steps, "seed": seed, "prompt": prompt}


def frame_dimensions(source_size: tuple[int, int], aspect: str, quality: str) -> tuple[int, int]:
    if aspect == ORIGINAL_ASPECT:
        if quality == ORIGINAL_SIZE:
            return source_size
        return original_dimensions(*source_size, quality)
    if aspect not in ASPECTS or quality == ORIGINAL_SIZE:
        raise ValueError("Choose a supported aspect ratio and size.")
    return dimensions(aspect, quality)


def _frame_settings(steps: int, seed: int, prompt: str, source_size: tuple[int, int],
                    aspect: str, quality: str, transparent: bool, true_cfg_scale: float,
                    negative_prompt: str | None) -> dict:
    width, height = frame_dimensions(source_size, aspect, quality)
    if not isinstance(steps, int) or not 1 <= steps <= 80:
        raise ValueError("Steps must be from 1 to 80.")
    prompt = (prompt or "").strip()
    if not prompt or len(prompt) > 4000:
        raise ValueError("Enter a prompt of up to 4,000 characters.")
    if seed == -1:
        seed = secrets.randbelow(2**32)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be -1 or from 0 to 4,294,967,295.")
    if isinstance(true_cfg_scale, bool) or not isinstance(true_cfg_scale, (int, float)) or not 1 <= true_cfg_scale <= 10:
        raise ValueError("CFG scale must be from 1.0 to 10.0.")
    if negative_prompt is None:
        negative_prompt = ""
    if not isinstance(negative_prompt, str) or len(negative_prompt.strip()) > 4000:
        raise ValueError("Keep the negative prompt under 4,000 characters.")
    return {
        "aspect": aspect, "quality": quality, "width": width, "height": height,
        "steps": steps, "seed": seed, "prompt": prompt, "transparent": bool(transparent),
        "true_cfg_scale": float(true_cfg_scale), "negative_prompt": negative_prompt.strip(),
    }


def preview_frames(job_id: str, processed: bool = False, limit: int = 12) -> list[tuple[str, str]]:
    job = load_job(job_id)
    directory = _job_dir(job_id) / (f"{_processed_name(job)}_frames" if processed else "source_frames")
    frames = numbered_frames(directory)
    if not frames:
        return []
    indices = sorted({round(i * (len(frames) - 1) / min(limit - 1, len(frames) - 1)) for i in range(min(limit, len(frames)))}) if len(frames) > 1 else [0]
    return [(str(frames[index]), f"Frame {int(frames[index].stem) + 1} / {job['frame_count']}") for index in indices]


def zip_frames(job_id: str, processed: bool = False) -> Path:
    job = load_job(job_id)
    name = f"{_processed_name(job)}_frames" if processed else "source_frames"
    directory = _job_dir(job_id) / name
    frames = numbered_frames(directory)
    if not frames:
        raise ValueError("No frames are available for download yet.")
    archive = _job_dir(job_id) / f"{name}.zip"
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for frame in frames:
            bundle.write(frame, frame.name)
    temporary.replace(archive)
    return archive


def _encode(job: dict) -> Path:
    directory = _job_dir(job["id"])
    name = _processed_name(job)
    output = directory / f"{name}.mp4"
    partial = directory / f"{name}.partial.mp4"
    fps = job["fps"]
    duration = job["frame_count"] / fps
    # The optional audio map also handles sources without an audio stream.
    audio_seek = ["-ss", str(job["start_seconds"])] if job["start_seconds"] else []
    _ffmpeg(
        "-framerate", str(fps), "-start_number", "0", "-i", str(directory / f"{name}_frames" / "%06d.png"),
        *audio_seek, "-i", str(directory / job["source_file"]),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", str(duration), "-movflags", "+faststart", str(partial),
    )
    partial.replace(output)
    return output


def _record_frame_timing(job_id: str, frame_number: int, timings: dict[str, float]) -> None:
    entry = {"frame": frame_number, "recorded_at": datetime.now(timezone.utc).isoformat(),
             **{key: round(value, 3) for key, value in timings.items()}}
    try:
        with (_job_dir(job_id) / "timings.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry) + "\n")
    except OSError:
        log.warning("Could not save timing for job %s frame %d", job_id, frame_number, exc_info=True)
    log.info(
        "Job %s frame %d timing: between %.2fs, load %.2fs, input %.2fs, "
        "to first step %.2fs, remaining model %.2fs, frame prep %.2fs, "
        "PNG save %.2fs, total %.2fs",
        job_id, frame_number, timings.get("between_frames_seconds", 0),
        timings.get("pipeline_load_seconds", 0), timings.get("input_prepare_seconds", 0),
        timings.get("first_step_seconds", 0),
        timings.get("pipeline_seconds", 0) - timings.get("first_step_seconds", 0),
        timings["frame_prepare_seconds"], timings["png_save_seconds"],
        timings["frame_total_seconds"],
    )


def process_job(job_id: str, long_edge: int | None, steps: int, seed: int, prompt: str,
                on_progress: Progress | None = None, expected_mode: str | None = None,
                aspect: str = ORIGINAL_ASPECT, quality: str = ORIGINAL_SIZE,
                transparent: bool = False, true_cfg_scale: float = 1.0,
                negative_prompt: str | None = "", vram_preset: str = "base") -> Path:
    validate_vram_preset(vram_preset)
    job = load_job(job_id)
    if expected_mode is not None and job_mode(job) != expected_mode:
        raise ValueError("This job belongs to the other video tab.")
    if job["status"] not in ("extracted", "processing", "failed", "paused", "complete"):
        raise ValueError(f"Job cannot be processed while status is {job['status']}.")
    source_size = (job["source_width"], job["source_height"])
    new_settings = (
        _settings(long_edge, steps, seed, prompt, source_size) if long_edge is not None else
        _frame_settings(steps, seed, prompt, source_size, aspect, quality, transparent,
                        true_cfg_scale, negative_prompt)
    )
    new_settings["vram_preset"] = vram_preset
    if job["settings"] is not None:
        job["settings"].setdefault("vram_preset", "base")
    if job["settings"] is None:
        job["settings"] = new_settings
        _write_job(job)
    elif seed == -1:
        # Resume a randomized job using the seed chosen at its first run.
        new_settings["seed"] = job["settings"]["seed"]
    if new_settings != job["settings"]:
        raise ValueError("Job settings are locked. Load the saved settings to resume, or extract a new job.")

    settings = job["settings"]
    size = (settings["width"], settings["height"])
    inference_size = model_dimensions(*size)
    frames = numbered_frames(_job_dir(job_id) / "source_frames")
    if len(frames) != job["frame_count"] or any(int(frame.stem) != i for i, frame in enumerate(frames)):
        raise RuntimeError("Source frame sequence is incomplete.")
    destination = _job_dir(job_id) / f"{_processed_name(job)}_frames"
    destination.mkdir(exist_ok=True)
    job["status"] = "processing"
    job.pop("error", None)
    _write_job(job)
    started = time.monotonic()
    processed_now = 0
    previous_frame_finished = None
    try:
        for i, source in enumerate(frames):
            frame_started = time.perf_counter()
            target = destination / source.name
            processed_frame = not _valid_frame(target, size)
            if processed_frame:
                timings: dict[str, float] = {}
                if previous_frame_finished is not None:
                    timings["between_frames_seconds"] = frame_started - previous_frame_finished
                prompt_text = settings["prompt"]
                if job_mode(job) == "edit":
                    prompt_text = (
                        f"Edit this video frame with one targeted change: {prompt_text} "
                        "Keep all other elements unchanged. Preserve the original composition and camera angle."
                    )
                request = Request(
                    "edit", prompt_text, settings.get("aspect", ORIGINAL_ASPECT),
                    settings.get("quality", ORIGINAL_SIZE), settings["steps"], settings["seed"],
                    settings.get("transparent", False), (source,), *inference_size,
                    settings.get("true_cfg_scale", 1.0), settings.get("negative_prompt", ""),
                    vram_preset=settings["vram_preset"],
                )
                log.info("Job %s frame %d/%d: starting model inference", job_id, i + 1, len(frames))
                generated = infer(request, timings)
                inference_finished = time.perf_counter()
                if generated.size != inference_size:
                    raise RuntimeError(f"Qwen returned {generated.size} for frame {i + 1}; expected {inference_size}.")
                if generated.size != size:
                    generated = generated.resize(size, Image.Resampling.LANCZOS)
                if settings.get("transparent", False):
                    frame = generated.convert("RGBA") if "A" not in generated.getbands() else generated
                else:
                    with Image.open(source) as original:
                        frame = original.convert("RGB").resize(size, Image.Resampling.LANCZOS)
                    if "A" in generated.getbands():
                        frame.paste(generated.convert("RGBA"), mask=generated.getchannel("A"))
                    else:
                        frame.paste(generated.convert("RGB"))
                save_started = time.perf_counter()
                temporary = target.with_suffix(".tmp.png")
                frame.save(temporary, "PNG")
                temporary.replace(target)
                save_finished = time.perf_counter()
                timings["frame_prepare_seconds"] = save_started - inference_finished
                timings["png_save_seconds"] = save_finished - save_started
                processed_now += 1
            job["completed_frames"] = i + 1
            _write_job(job)
            if processed_frame:
                timings["frame_total_seconds"] = time.perf_counter() - frame_started
                _record_frame_timing(job_id, i + 1, timings)
            previous_frame_finished = time.perf_counter()
            elapsed = time.monotonic() - started
            remaining = ((elapsed / processed_now) * (len(frames) - i - 1)) if processed_now else None
            if on_progress:
                on_progress(i + 1, len(frames), remaining)
        job["status"] = "encoding"
        _write_job(job)
        result = _encode(job)
        job["status"] = "complete"
        job["video_file"] = result.name
        _write_job(job)
        return result
    except PauseRequested:
        job["status"] = "paused"
        _write_job(job)
        raise
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        _write_job(job)
        raise
