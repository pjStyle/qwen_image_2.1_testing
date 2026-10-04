"""Local web interface for Qwen Image 2.1."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import gradio as gr

from qwen_workspace.core import ASPECTS, ORIGINAL_ASPECT, ORIGINAL_SIZE, QUALITY_PIXELS, save_result, validate_request
from qwen_workspace.model import infer
from qwen_workspace.video import (
    DEFAULT_PROMPT, EDIT_PROMPT_HINT, PauseRequested, create_job, job_mode, load_job,
    frame_dimensions, preview_frames, process_job, require_job_mode, target_dimensions, zip_frames,
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
    true_cfg_scale: float = 1.0,
    negative_prompt: str = "",
    progress=gr.Progress(),
):
    # Gradio Gallery inputs are (filepath, caption) pairs; the request validator
    # and model pipeline consume only file paths. Keep gallery order intact.
    if references:
        references = [item[0] if isinstance(item, (tuple, list)) else item for item in references]
    try:
        request = validate_request(
            mode, prompt, aspect, quality, int(steps), int(seed), transparent, references,
            true_cfg_scale, negative_prompt,
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
        editing = prefix == "edit"
        aspect = gr.Dropdown(
            choices=([ORIGINAL_ASPECT] if editing else []) + list(ASPECTS),
            value=ORIGINAL_ASPECT if editing else "1:1",
            label="Aspect ratio",
        )
        quality = gr.Dropdown(
            choices=([ORIGINAL_SIZE] if editing else []) + list(QUALITY_PIXELS),
            value=ORIGINAL_SIZE if editing else "Standard (~1 MP)",
            label="Size",
        )
        if editing:
            aspect.change(
                lambda selected_aspect, selected_quality: (
                    "Standard (~1 MP)" if selected_aspect != ORIGINAL_ASPECT and selected_quality == ORIGINAL_SIZE
                    else selected_quality
                ),
                [aspect, quality], [quality], queue=False,
            )
            quality.change(
                lambda selected_quality, selected_aspect: (
                    ORIGINAL_ASPECT if selected_quality == ORIGINAL_SIZE else selected_aspect
                ),
                [quality, aspect], [aspect], queue=False,
            )
    with gr.Row():
        steps = gr.Slider(1, 80, value=40, step=1, label="Steps")
        seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
    transparent = gr.Checkbox(label="Request transparent background (RGBA)")
    with gr.Accordion("Advanced guidance", open=False):
        gr.Markdown(
            "Qwen Image 2.1 defaults to CFG 1.0 (no guidance). Set CFG above 1.0 to strengthen prompt "
            "adherence; negative prompts only take effect above 1.0. Guidance also increases processing time."
        )
        true_cfg_scale = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="CFG scale")
        negative_prompt = gr.Textbox(
            label="Negative prompt",
            lines=3,
            placeholder="Optional: describe what the result should avoid",
        )
    output = gr.Image(type="filepath", label="Result", interactive=False)
    status = gr.Markdown()
    return aspect, quality, steps, seed, transparent, true_cfg_scale, negative_prompt, output, status


def _video_status(job: dict, extra: str = "") -> str:
    folder = Path(__file__).resolve().parent / "outputs" / "video_jobs" / job["id"]
    result_dir = "edited_frames" if job_mode(job) == "edit" else "upscaled_frames"
    message = (
        f"**Job {job['id']}** · {job['status']} · "
        f"{job.get('completed_frames', 0)}/{job['frame_count']} frames · "
        f"{job['fps']} FPS · {job['output_duration_seconds']:.1f} s output\n\n"
        f"Source frames: `{folder / 'source_frames'}`  \n"
        f"Qwen frames: `{folder / result_dir}`"
    )
    if extra:
        message += f"\n\n{extra}"
    return message


def video_size_note(job_id: str, aspect: str, quality: str, legacy_long_edge: int | None = None) -> str:
    if not job_id:
        return "Extract or load a video to see its output dimensions."
    try:
        job = load_job(job_id)
        source = (job["source_width"], job["source_height"])
        width, height = (target_dimensions(*source, legacy_long_edge) if legacy_long_edge is not None
                         else frame_dimensions(source, aspect, quality))
        legacy = " (saved long-edge setting)" if legacy_long_edge is not None else ""
        return f"Output: **{width}×{height}** from {source[0]}×{source[1]}{legacy}."
    except (ValueError, KeyError):
        return "Extract or load a valid video job to see its output dimensions."


def extract_video(video_path: str | None, fps: float, start: float, end: float | None,
                  aspect: str, quality: str, mode: str, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    progress(0, desc="Copying video and extracting frames")
    try:
        job = create_job(video_path, int(fps), float(start or 0), float(end) if end is not None else None, mode=mode)
    except Exception as exc:
        log.exception("Video extraction failed")
        raise gr.Error(str(exc)) from exc
    progress(1, desc="Frames ready")
    return job["id"], preview_frames(job["id"]), [], None, _video_status(
        job, "Review the source frames, then set Qwen options and start the batch."
    ), video_size_note(job["id"], aspect, quality), None


def load_video_job(job_id: str, mode: str):
    try:
        job = require_job_mode(job_id, mode)
        settings = job.get("settings") or {}
        video_name = "edited.mp4" if mode == "edit" else "upscaled.mp4"
        video_path = str(Path(__file__).resolve().parent / "outputs" / "video_jobs" / job_id / video_name)
        if not Path(video_path).is_file():
            video_path = None
        return (
            preview_frames(job_id), preview_frames(job_id, processed=True), video_path,
            _video_status(job, "Resume with the saved settings if this job is incomplete."),
            settings.get("aspect", ORIGINAL_ASPECT), settings.get("quality", ORIGINAL_SIZE),
            settings.get("steps", 40),
            settings.get("seed", -1), settings.get("prompt", "" if mode == "edit" else DEFAULT_PROMPT),
            settings.get("transparent", False), settings.get("true_cfg_scale", 1.0),
            settings.get("negative_prompt", ""),
            video_size_note(job_id, settings.get("aspect", ORIGINAL_ASPECT),
                            settings.get("quality", ORIGINAL_SIZE), settings.get("long_edge")),
            settings.get("long_edge"),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def run_video_job(job_id: str, aspect: str, quality: str, steps: float, seed: float,
                  prompt: str, transparent: bool, true_cfg_scale: float, negative_prompt: str,
                  legacy_long_edge: int | None, mode: str, progress=gr.Progress()):
    if not job_id:
        raise gr.Error("Extract a video or load a saved job first.")
    try:
        require_job_mode(job_id, mode)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    event = threading.Event()
    _pause_flags[job_id] = event

    def on_frame(done: int, total: int, remaining: float | None) -> None:
        estimate = f" · about {remaining / 60:.1f} min remaining" if remaining is not None else ""
        progress(done / total, desc=f"Qwen frame {done}/{total}{estimate}")
        if event.is_set():
            raise PauseRequested()

    try:
        result = process_job(
            job_id, legacy_long_edge, int(steps), int(seed), prompt,
            on_progress=on_frame, expected_mode=mode, aspect=aspect, quality=quality,
            transparent=transparent, true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
        )
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


def download_frames(job_id: str, processed: bool, mode: str):
    try:
        require_job_mode(job_id, mode)
        return str(zip_frames(job_id, processed))
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def video_tab(mode: str) -> None:
    editing = mode == "edit"
    mode_state = gr.State(mode)
    if editing:
        gr.Markdown(
            "**Experimental:** Apply one edit prompt to every extracted frame, then reassemble the video. "
            "Each frame is edited independently, so the changed element may shift or flicker. "
            "Try a short clip at a low FPS first."
        )
    else:
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
        aspect = gr.Dropdown([ORIGINAL_ASPECT, *ASPECTS], value=ORIGINAL_ASPECT, label="Output frame aspect ratio")
        quality = gr.Dropdown([ORIGINAL_SIZE, *QUALITY_PIXELS], value=ORIGINAL_SIZE, label="Output frame size")
    aspect.change(
        lambda selected_aspect, selected_quality: (
            "Standard (~1 MP)" if selected_aspect != ORIGINAL_ASPECT and selected_quality == ORIGINAL_SIZE
            else selected_quality
        ), [aspect, quality], [quality], queue=False,
    )
    quality.change(
        lambda selected_quality, selected_aspect: (
            ORIGINAL_ASPECT if selected_quality == ORIGINAL_SIZE else selected_aspect
        ), [quality, aspect], [aspect], queue=False,
    )
    with gr.Row():
        video_steps = gr.Slider(1, 80, value=40, step=1, label="Qwen steps per frame")
        video_seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
    transparent = gr.Checkbox(label="Request transparent output frames (RGBA)")
    with gr.Accordion("Advanced guidance", open=False):
        gr.Markdown("CFG defaults to 1.0. Negative prompts take effect only above 1.0.")
        cfg = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="CFG scale")
        negative_prompt = gr.Textbox(label="Negative prompt", lines=3,
                                     placeholder="Optional: describe what the result should avoid")
    legacy_long_edge = gr.State(None)
    size_note = gr.Markdown("Extract or load a video to see its output dimensions.")
    video_prompt = gr.Textbox(
        value="" if editing else DEFAULT_PROMPT,
        label="Change to make in every frame" if editing else "Preservation prompt",
        placeholder=EDIT_PROMPT_HINT if editing else None,
        lines=3,
    )
    with gr.Row():
        process_button = gr.Button("2. Start / resume Qwen batch", variant="primary")
        pause_button = gr.Button("Pause after current frame")
    result_gallery = gr.Gallery(label="Edited frame samples" if editing else "Upscaled frame samples", columns=4, object_fit="contain")
    completed_video = gr.Video(label="Completed MP4", interactive=False)
    video_status = gr.Markdown()
    with gr.Row():
        source_zip_button = gr.Button("Download source frames ZIP")
        result_zip_button = gr.Button("Download edited frames ZIP" if editing else "Download Qwen frames ZIP")
    source_zip = gr.File(label="Source frames ZIP", interactive=False)
    result_zip = gr.File(label="Edited frames ZIP" if editing else "Qwen frames ZIP", interactive=False)
    extract_button.click(
        extract_video, [video_input, fps, start, end, aspect, quality, mode_state],
        [job_id, source_gallery, result_gallery, completed_video, video_status, size_note, legacy_long_edge],
    )
    load_button.click(
        load_video_job, [job_id, mode_state],
        [source_gallery, result_gallery, completed_video, video_status,
         aspect, quality, video_steps, video_seed, video_prompt, transparent, cfg,
         negative_prompt, size_note, legacy_long_edge],
    )
    aspect.change(video_size_note, [job_id, aspect, quality, legacy_long_edge], [size_note], queue=False)
    quality.change(video_size_note, [job_id, aspect, quality, legacy_long_edge], [size_note], queue=False)
    process_button.click(
        run_video_job, [job_id, aspect, quality, video_steps, video_seed, video_prompt,
                        transparent, cfg, negative_prompt, legacy_long_edge, mode_state],
        [result_gallery, completed_video, video_status],
        concurrency_limit=1, concurrency_id="gpu",
    )
    pause_button.click(pause_video_job, [job_id], [video_status], queue=False)
    source_zip_button.click(lambda value, selected: download_frames(value, False, selected), [job_id, mode_state], [source_zip])
    result_zip_button.click(lambda value, selected: download_frames(value, True, selected), [job_id, mode_state], [result_zip])


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Qwen Image 2.1 Local") as demo:
        gr.Markdown(
            "# Qwen Image 2.1 Local\n"
            "Generate images, edit images, upscale video, or experimentally edit a video. First use downloads about 33 GB of model files; "
            "the console shows download progress. Small size is quickest; Edit preserves input resolution by default."
        )
        with gr.Tabs():
            with gr.Tab("Generate"):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the image to create")
                aspect, quality, steps, seed, transparent, cfg, negative_prompt, output, status = controls("generate")
                button = gr.Button("Generate image", variant="primary")
                button.click(
                    run,
                    inputs=[
                        gr.State("generate"), prompt, gr.State(None), aspect, quality, steps, seed,
                        transparent, cfg, negative_prompt,
                    ],
                    outputs=[output, status],
                    concurrency_limit=1,
                    concurrency_id="gpu",
                )
            with gr.Tab("Edit"):
                edit_prompt = gr.Textbox(label="Edit prompt", lines=4, placeholder="Describe how to change or combine the references")
                references = gr.Gallery(
                    label="Reference images (1–10, in upload order)",
                    columns=4,
                    object_fit="contain",
                    interactive=True,
                    sources=["upload"],
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                    type="filepath",
                )
                (
                    e_aspect, e_quality, e_steps, e_seed, e_transparent, e_cfg, e_negative_prompt,
                    e_output, e_status,
                ) = controls("edit")
                edit_button = gr.Button("Edit image", variant="primary")
                edit_button.click(
                    run,
                    inputs=[
                        gr.State("edit"), edit_prompt, references, e_aspect, e_quality, e_steps,
                        e_seed, e_transparent, e_cfg, e_negative_prompt,
                    ],
                    outputs=[e_output, e_status],
                    concurrency_limit=1,
                    concurrency_id="gpu",
                )
            with gr.Tab("Video"):
                video_tab("upscale")
            with gr.Tab("Video Edit (experimental)"):
                video_tab("edit")
        gr.Markdown("Results and their settings are saved in the `outputs` folder. A random seed is shown after each run.")
    return demo


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(__import__("pathlib").Path(__file__).resolve().parent / ".hf-cache"))
    host = os.environ.get("QWEN_HOST", "127.0.0.1")
    port = int(os.environ.get("QWEN_PORT", "7860"))
    password = os.environ.get("QWEN_AUTH_PASSWORD")
    if host not in {"127.0.0.1", "localhost", "::1"} and not password:
        raise RuntimeError("QWEN_AUTH_PASSWORD is required when QWEN_HOST is not localhost.")
    auth = (os.environ.get("QWEN_AUTH_USER", "qwen"), password) if password else None
    from qwen_workspace.check_environment import main as check_environment

    check_environment()
    log.info("Starting interface on %s:%s", host, port)
    build_app().queue(max_size=1, default_concurrency_limit=1).launch(
        server_name=host, server_port=port, share=False,
        inbrowser=host in {"127.0.0.1", "localhost", "::1"}, auth=auth,
    )
