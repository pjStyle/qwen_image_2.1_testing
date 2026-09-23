"""Local web interface for Qwen Image 2.1."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import gradio as gr

from qwen_workspace.core import ASPECTS, QUALITY_PIXELS, save_result, validate_request
from qwen_workspace.model import infer
from qwen_workspace.video import (
    DEFAULT_PROMPT, PauseRequested, create_job, load_job, preview_frames, process_job,
    target_dimensions, zip_frames,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
_pause_flags: dict[str, threading.Event] = {}


def run(
    mode: str,
    prompt: str,
    references: list[str] | None,
    aspect: str,
    quality: str,
    steps: float,
    seed: float,
    transparent: bool,
    progress=gr.Progress(),
):
    try:
        request = validate_request(
            mode, prompt, aspect, quality, int(steps), int(seed), transparent, references
        )
    except (TypeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    try:
        progress(0, desc="Checking model download and loading pipeline. First run can take a while.")
        image = infer(request)
        progress(0.9, desc="Saving PNG and settings")
        image_path, _ = save_result(image, request)
        progress(1, desc="Done")
        alpha = " with alpha" if "A" in image.getbands() else ""
        return str(image_path), f"Saved **{image_path.name}**{alpha} · seed **{request.seed}**"
    except Exception as exc:
        log.exception("Generation failed")
        try:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                raise gr.Error(
                    "GPU memory ran out. Try Small size, fewer reference images, or close other GPU apps. Your prompt and settings are still in the form."
                ) from exc
        except ImportError:
            pass
        raise gr.Error(f"Generation failed: {exc}") from exc


def controls(prefix: str):
    with gr.Row():
        aspect = gr.Dropdown(choices=list(ASPECTS), value="1:1", label="Aspect ratio")
        quality = gr.Dropdown(
            choices=list(QUALITY_PIXELS),
            value="Standard (~1 MP)",
            label="Size",
        )
    with gr.Row():
        steps = gr.Slider(1, 80, value=40, step=1, label="Steps")
        seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
    transparent = gr.Checkbox(label="Request transparent background (RGBA)")
    output = gr.Image(type="filepath", label="Result", interactive=False)
    status = gr.Markdown()
    return aspect, quality, steps, seed, transparent, output, status


def _video_status(job: dict, extra: str = "") -> str:
    folder = Path(__file__).resolve().parent / "outputs" / "video_jobs" / job["id"]
    message = (
        f"**Job {job['id']}** · {job['status']} · "
        f"{job.get('completed_frames', 0)}/{job['frame_count']} frames · "
        f"{job['fps']} FPS · {job['output_duration_seconds']:.1f} s output\n\n"
        f"Source frames: `{folder / 'source_frames'}`  \n"
        f"Qwen frames: `{folder / 'upscaled_frames'}`"
    )
    if extra:
        message += f"\n\n{extra}"
    return message


def video_size_note(job_id: str, long_edge: float) -> str:
    if not job_id:
        return "Extract or load a video to see its output dimensions."
    try:
        job = load_job(job_id)
        width, height = target_dimensions(job["source_width"], job["source_height"], int(long_edge))
        source = (job["source_width"], job["source_height"])
        relation = "downscaling" if int(long_edge) < max(source) else "upscaling" if int(long_edge) > max(source) else "same long edge"
        return f"Output: **{width}×{height}** from {source[0]}×{source[1]} ({relation})."
    except (ValueError, KeyError):
        return "Extract or load a valid video job to see its output dimensions."


def extract_video(video_path: str | None, fps: float, start: float, end: float | None, long_edge: float, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    progress(0, desc="Copying video and extracting frames")
    try:
        job = create_job(video_path, int(fps), float(start or 0), float(end) if end is not None else None)
    except Exception as exc:
        log.exception("Video extraction failed")
        raise gr.Error(str(exc)) from exc
    progress(1, desc="Frames ready")
    return job["id"], preview_frames(job["id"]), [], None, _video_status(
        job, "Review the source frames, then set Qwen options and start the batch."
    ), video_size_note(job["id"], long_edge)


def load_video_job(job_id: str):
    try:
        job = load_job(job_id)
        settings = job.get("settings") or {}
        video_path = str(Path(__file__).resolve().parent / "outputs" / "video_jobs" / job_id / "upscaled.mp4")
        if not Path(video_path).is_file():
            video_path = None
        return (
            preview_frames(job_id), preview_frames(job_id, processed=True), video_path,
            _video_status(job, "Resume with the saved settings if this job is incomplete."),
            settings.get("long_edge", 1024), settings.get("steps", 40),
            settings.get("seed", -1), settings.get("prompt", DEFAULT_PROMPT),
            video_size_note(job_id, settings.get("long_edge", 1024)),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def run_video_job(job_id: str, long_edge: float, steps: float, seed: float, prompt: str, progress=gr.Progress()):
    if not job_id:
        raise gr.Error("Extract a video or load a saved job first.")
    event = threading.Event()
    _pause_flags[job_id] = event

    def on_frame(done: int, total: int, remaining: float | None) -> None:
        estimate = f" · about {remaining / 60:.1f} min remaining" if remaining is not None else ""
        progress(done / total, desc=f"Qwen frame {done}/{total}{estimate}")
        if event.is_set():
            raise PauseRequested()

    try:
        result = process_job(job_id, int(long_edge), int(steps), int(seed), prompt, on_progress=on_frame)
        job = load_job(job_id)
        return preview_frames(job_id, processed=True), str(result), _video_status(job, "Video ready to preview or download.")
    except PauseRequested:
        job = load_job(job_id)
        return preview_frames(job_id, processed=True), None, _video_status(job, "Paused after the current frame. Press Start / resume to continue.")
    except Exception as exc:
        log.exception("Video batch failed")
        job = load_job(job_id)
        return preview_frames(job_id, processed=True), None, _video_status(job, f"**Stopped:** {exc}. Fix the issue and press Start / resume.")
    finally:
        _pause_flags.pop(job_id, None)


def pause_video_job(job_id: str) -> str:
    event = _pause_flags.get(job_id)
    if event is None:
        return "No active video batch for this job."
    event.set()
    return "Pause requested. The current frame will finish and the batch will stop."


def download_frames(job_id: str, processed: bool):
    try:
        return str(zip_frames(job_id, processed))
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Qwen Image 2.1 Local") as demo:
        gr.Markdown(
            "# Qwen Image 2.1 Local\n"
            "Generate images, edit images, or upscale video frames on this PC. First use downloads about 33 GB of model files; "
            "the console shows download progress. Small size is quickest; Standard is the default for 16 GB VRAM."
        )
        with gr.Tabs():
            with gr.Tab("Generate"):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the image to create")
                aspect, quality, steps, seed, transparent, output, status = controls("generate")
                button = gr.Button("Generate image", variant="primary")
                button.click(
                    run,
                    inputs=[gr.State("generate"), prompt, gr.State(None), aspect, quality, steps, seed, transparent],
                    outputs=[output, status],
                    concurrency_limit=1,
                    concurrency_id="gpu",
                )
            with gr.Tab("Edit"):
                edit_prompt = gr.Textbox(label="Edit prompt", lines=4, placeholder="Describe how to change or combine the references")
                references = gr.File(
                    label="Reference images (1–10, in upload order)",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                    type="filepath",
                )
                e_aspect, e_quality, e_steps, e_seed, e_transparent, e_output, e_status = controls("edit")
                edit_button = gr.Button("Edit image", variant="primary")
                edit_button.click(
                    run,
                    inputs=[gr.State("edit"), edit_prompt, references, e_aspect, e_quality, e_steps, e_seed, e_transparent],
                    outputs=[e_output, e_status],
                    concurrency_limit=1,
                    concurrency_id="gpu",
                )
            with gr.Tab("Video"):
                gr.Markdown(
                    "Extract frames first, review them, then process the batch through Qwen. "
                    "Each frame is edited independently; details may change or flicker. "
                    "At 10 FPS, even a short clip can take a long time."
                )
                video_input = gr.Video(label="Source video", sources=["upload"], format=None)
                with gr.Row():
                    fps = gr.Slider(1, 60, value=10, step=1, label="Output FPS / frames extracted per second")
                    start = gr.Number(value=0, minimum=0, label="Start time (seconds)")
                    end = gr.Number(value=None, minimum=0, label="End time (blank = full clip)")
                extract_button = gr.Button("1. Extract frames")
                job_id = gr.Textbox(label="Job ID (keep this to resume after restarting)")
                load_button = gr.Button("Load saved job")
                source_gallery = gr.Gallery(label="Source frame samples", columns=4, object_fit="contain")
                with gr.Row():
                    long_edge = gr.Slider(256, 2048, value=1024, step=16, label="Output long edge (pixels)")
                    video_steps = gr.Slider(1, 80, value=40, step=1, label="Qwen steps per frame")
                    video_seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                size_note = gr.Markdown("Extract or load a video to see its output dimensions.")
                video_prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Preservation prompt", lines=3)
                with gr.Row():
                    process_button = gr.Button("2. Start / resume Qwen batch", variant="primary")
                    pause_button = gr.Button("Pause after current frame")
                result_gallery = gr.Gallery(label="Upscaled frame samples", columns=4, object_fit="contain")
                completed_video = gr.Video(label="Completed MP4", interactive=False)
                video_status = gr.Markdown()
                with gr.Row():
                    source_zip_button = gr.Button("Download source frames ZIP")
                    result_zip_button = gr.Button("Download Qwen frames ZIP")
                source_zip = gr.File(label="Source frames ZIP", interactive=False)
                result_zip = gr.File(label="Qwen frames ZIP", interactive=False)
                extract_button.click(
                    extract_video, [video_input, fps, start, end, long_edge],
                    [job_id, source_gallery, result_gallery, completed_video, video_status, size_note],
                )
                load_button.click(
                    load_video_job, [job_id],
                    [source_gallery, result_gallery, completed_video, video_status,
                     long_edge, video_steps, video_seed, video_prompt, size_note],
                )
                long_edge.change(video_size_note, [job_id, long_edge], [size_note], queue=False)
                process_button.click(
                    run_video_job, [job_id, long_edge, video_steps, video_seed, video_prompt],
                    [result_gallery, completed_video, video_status],
                    concurrency_limit=1, concurrency_id="gpu",
                )
                pause_button.click(pause_video_job, [job_id], [video_status], queue=False)
                source_zip_button.click(lambda value: download_frames(value, False), [job_id], [source_zip])
                result_zip_button.click(lambda value: download_frames(value, True), [job_id], [result_zip])
        gr.Markdown("Results and their settings are saved in the `outputs` folder. A random seed is shown after each run.")
    return demo


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(__import__("pathlib").Path(__file__).resolve().parent / ".hf-cache"))
    from qwen_workspace.check_environment import main as check_environment

    check_environment()
    log.info("Starting local interface at http://127.0.0.1:7860")
    build_app().queue(max_size=1, default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True
    )
