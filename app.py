"""Local web interface for Qwen Image 2.1."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path

import gradio as gr

from qwen_workspace.core import (
    ASPECTS, ORIGINAL_ASPECT, POSE_ASPECT, ORIGINAL_SIZE, QUALITY_PIXELS,
    REFERENCE_QUALITY_PIXELS, VRAM_PRESETS, save_result, validate_request,
)
from qwen_workspace import model, pose_analyzer, pose_references
from qwen_workspace.model import infer
from qwen_workspace.metrics import PeakRssSampler
from qwen_workspace.perspectives import (
    PERSPECTIVES, create_batch, generate_batch, prepare_requests, zip_batch,
)
from qwen_workspace import poses
from qwen_workspace.video import (
    DEFAULT_PROMPT, EDIT_PROMPT_HINT, PauseRequested, create_job, job_mode, load_job,
    frame_dimensions, preview_frames, process_job, require_job_mode, target_dimensions, zip_frames,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
_pause_flags: dict[str, threading.Event] = {}
_cancel_event = threading.Event()


def cancel_current_job() -> str:
    _cancel_event.set()
    return "Cancellation requested. The current model step will stop shortly; completed results are kept."


def reset_model() -> str:
    pose_analyzer.reset()
    model.reset_pipeline()
    return "Model unloaded and CUDA cache cleared. The next job will reload the model."


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
    reference_quality: str = ORIGINAL_SIZE,
    progress=gr.Progress(),
    vram_preset: str = "base",
):
    generation_started = time.perf_counter()
    # Gradio Gallery inputs are (filepath, caption) pairs; the request validator
    # and model pipeline consume only file paths. Keep gallery order intact.
    if references:
        references = [item[0] if isinstance(item, (tuple, list)) else item for item in references]
    try:
        request = validate_request(
            mode, prompt, aspect, quality, int(steps), int(seed), transparent, references,
            true_cfg_scale, negative_prompt, reference_quality, vram_preset,
        )
    except (TypeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    try:
        _cancel_event.clear()
        model.set_cancel_event(_cancel_event)
        progress(0, desc="Checking model download and loading pipeline. First run can take a while.")
        timings: dict[str, float | int] = {}
        with PeakRssSampler() as ram_sampler:
            image = infer(request, timings)
            progress(0.9, desc="Saving PNG and settings")
            image_path, _ = save_result(image, request)
        elapsed = time.perf_counter() - generation_started
        resource_notes = []
        allocated = timings.get("peak_vram_allocated_bytes")
        reserved = timings.get("peak_vram_reserved_bytes")
        if allocated is not None and reserved is not None:
            mib = 1024**2
            resource_notes.append(
                f"peak VRAM allocated/reserved {allocated / mib:.0f}/{reserved / mib:.0f} MiB"
            )
        if ram_sampler.peak_bytes is not None:
            resource_notes.append(f"sampled peak process RAM {ram_sampler.peak_bytes / 1024**2:.0f} MiB")
        else:
            resource_notes.append("peak process RAM unavailable")
        log.info("Generation completed in %.2f seconds | %s", elapsed, " | ".join(resource_notes))
        progress(1, desc="Done")
        alpha = " with alpha" if "A" in image.getbands() else ""
        return str(image_path), (
            f"Saved **{image_path.name}**{alpha} · {VRAM_PRESETS[request.vram_preset]} · "
            f"seed **{request.seed}** · {elapsed:.1f} s"
        )
    except Exception as exc:
        log.exception("Generation failed after %.2f seconds", time.perf_counter() - generation_started)
        if isinstance(exc, model.InferenceCancelled):
            raise gr.Error("Generation cancelled. Your prompt and settings are still in the form.") from exc
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
    finally:
        model.set_cancel_event(None)


def controls(prefix: str, *, include_output: bool = True):
    with gr.Row():
        editing = prefix in {"edit", "perspectives", "poses", "pose_references"}
        native_aspects = ([POSE_ASPECT, ORIGINAL_ASPECT] if prefix == "pose_references"
                          else [ORIGINAL_ASPECT] if editing else [])
        default_aspect = native_aspects[0] if editing else "1:1"
        aspect = gr.Dropdown(
            choices=native_aspects + list(ASPECTS),
            value=default_aspect,
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
                    "Standard (~1 MP)" if selected_aspect not in native_aspects and selected_quality == ORIGINAL_SIZE
                    else selected_quality
                ),
                [aspect, quality], [quality], queue=False,
            )
            quality.change(
                lambda selected_quality, selected_aspect: (
                    (selected_aspect if selected_aspect in native_aspects else default_aspect)
                    if selected_quality == ORIGINAL_SIZE else selected_aspect
                ),
                [quality, aspect], [aspect], queue=False,
            )
    with gr.Row():
        steps = gr.Slider(1, 80, value=40, step=1, label="Steps")
        seed = gr.Number(value=42 if prefix in {"perspectives", "poses", "pose_references"} else -1, precision=0, label="Seed (-1 for random)")
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
    output = gr.Image(type="filepath", label="Result", interactive=False) if include_output else None
    status = gr.Markdown() if include_output else None
    return aspect, quality, steps, seed, transparent, true_cfg_scale, negative_prompt, output, status


def run_perspectives(
    references, selected, instructions, aspect, quality, steps, seed, transparent, cfg,
    negative_prompt, reference_quality, vram_preset, three_quarter_prompt, half_body_prompt,
    full_body_prompt, birds_eye_prompt, worms_eye_prompt, face_closeup_prompt,
    use_perspective_settings=True,
    progress=gr.Progress(),
):
    references = [item[0] if isinstance(item, (tuple, list)) else item for item in (references or [])]
    prompts = dict(zip(PERSPECTIVES, [
        three_quarter_prompt, half_body_prompt, full_body_prompt,
        birds_eye_prompt, worms_eye_prompt, face_closeup_prompt,
    ]))
    try:
        requests = prepare_requests(
            references, selected, instructions, prompts, aspect=aspect, quality=quality,
            steps=int(steps), seed=int(seed), transparent=transparent, true_cfg_scale=cfg,
            negative_prompt=negative_prompt, reference_quality=reference_quality, vram_preset=vram_preset,
            use_perspective_settings=use_perspective_settings,
        )
    except (TypeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    gallery = []
    batch = None
    iterator = None
    try:
        _cancel_event.clear()
        model.set_cancel_event(_cancel_event)
        batch = create_batch(requests)
        total = len(requests)
        used_seed = requests[0][1].seed
        yield list(gallery), None, f"Starting {total} perspectives · seed **{used_seed}** · **{VRAM_PRESETS[vram_preset]}**"
        iterator = generate_batch(batch)
        for index, (key, _) in enumerate(requests):
            progress(index / total, desc=f"Perspective {index + 1}/{total}: {PERSPECTIVES[key].label}")
            result = next(iterator)
            gallery.append((str(batch.folder / result["image"]), result["label"]))
            yield list(gallery), None, (
                f"Completed **{index + 1}/{total}** · seed **{used_seed}** · "
                f"{result['label']} in {result['total_seconds']:.1f}s"
            )
        # Exhaust the generator so its manifest records completion before creating the ZIP.
        next(iterator, None)
        archive = zip_batch(batch)
        progress(1, desc="Perspectives ready")
        yield list(gallery), str(archive), (
            f"Completed **{total}/{total}** · seed **{used_seed}** · **{VRAM_PRESETS[vram_preset]}**\n\n"
            f"Saved to `outputs/perspectives/{batch.folder.name}`. ZIP includes images, prompts, settings and references."
        )
    except Exception as exc:
        log.exception("Perspective generation failed")
        archive = None
        if batch is not None:
            try:
                archive = str(zip_batch(batch))
            except Exception:
                log.exception("Could not package partial perspective batch")
        yield list(gallery), archive, f"**Stopped after {len(gallery)}/{len(requests)} perspectives:** {exc}. Completed images remain saved."
    finally:
        model.set_cancel_event(None)
        if iterator is not None:
            iterator.close()


def perspectives_tab(vram_preset) -> None:
    gr.Markdown(
        "Upload a subject and choose its views. Optional extra references should show the same subject. "
        "Each view uses the uploaded originals and the same seed. The prompts preserve the subject's "
        "appearance and setting. Use the VRAM preset above; Small size is useful for quick trials."
    )
    references = gr.Gallery(
        label="Reference images (1–10, first image sets original output size)", columns=4,
        object_fit="contain", interactive=True, sources=["upload"], type="filepath",
        file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    )
    selected = gr.CheckboxGroup(
        choices=[(item.label, key) for key, item in PERSPECTIVES.items()],
        value=list(PERSPECTIVES), label="Perspectives to generate",
    )
    instructions = gr.Textbox(
        label="Additional instructions for every view (optional)", lines=2,
        placeholder="For example: keep the jacket's pattern and the original expression",
    )
    with gr.Accordion("Perspective prompts", open=False):
        prompt_inputs = []
        for item in PERSPECTIVES.values():
            prompt_inputs.append(gr.Textbox(label=item.label, value=item.prompt, lines=4))
            gr.Textbox(
                label=f"Saved negative prompt · {item.label} · CFG {item.true_cfg_scale:g}",
                value=item.negative_prompt, lines=2, interactive=False,
            )
        reset = gr.Button("Reset prompts")
        reset.click(lambda: [item.prompt for item in PERSPECTIVES.values()], outputs=prompt_inputs, queue=False)
    reference_quality = gr.Dropdown(
        choices=[ORIGINAL_SIZE, *REFERENCE_QUALITY_PIXELS], value=ORIGINAL_SIZE,
        label="Reference image size",
    )
    aspect, quality, steps, seed, transparent, cfg, negative, _, _ = controls("perspectives", include_output=False)
    use_perspective_settings = gr.Checkbox(
        value=True, label="Use saved per-view negative prompts and CFG",
        info="Uses CFG 2 for body crops, full body and worm's-eye; CFG 3 for bird's-eye. "
             "Face close-up keeps CFG 1 and a blank negative prompt. "
             "Turn off to use the shared CFG and negative prompt controls above.",
    )
    button = gr.Button("Generate selected perspectives", variant="primary")
    gallery = gr.Gallery(label="Generated perspectives", columns=3, object_fit="contain", interactive=False)
    archive = gr.File(label="Download perspectives ZIP", interactive=False)
    status = gr.Markdown()
    button.click(
        run_perspectives,
        inputs=[references, selected, instructions, aspect, quality, steps, seed, transparent,
                cfg, negative, reference_quality, vram_preset, *prompt_inputs, use_perspective_settings],
        outputs=[gallery, archive, status], concurrency_limit=1, concurrency_id="gpu",
        show_progress="minimal",
    )


def run_poses(
    references, selected, instructions, aspect, quality, steps, seed, transparent, cfg,
    negative_prompt, reference_quality, vram_preset, wall_lean_prompt, victory_cheer_prompt,
    flirty_smile_prompt, confident_hero_prompt, curious_thinker_prompt, surprised_reaction_prompt,
    use_pose_settings=True,
    progress=gr.Progress(),
):
    references = [item[0] if isinstance(item, (tuple, list)) else item for item in (references or [])]
    prompts = dict(zip(poses.POSES, [
        wall_lean_prompt, victory_cheer_prompt, flirty_smile_prompt,
        confident_hero_prompt, curious_thinker_prompt, surprised_reaction_prompt,
    ]))
    try:
        requests = poses.prepare_requests(
            references, selected, instructions, prompts, aspect=aspect, quality=quality,
            steps=int(steps), seed=int(seed), transparent=transparent, true_cfg_scale=cfg,
            negative_prompt=negative_prompt, reference_quality=reference_quality, vram_preset=vram_preset,
            use_pose_settings=use_pose_settings,
        )
    except (TypeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    gallery = []
    batch = None
    iterator = None
    try:
        _cancel_event.clear()
        model.set_cancel_event(_cancel_event)
        batch = poses.create_batch(requests)
        total = len(requests)
        used_seed = requests[0][1].seed
        yield list(gallery), None, f"Starting {total} poses and expressions · seed **{used_seed}** · **{VRAM_PRESETS[vram_preset]}**"
        iterator = poses.generate_batch(batch)
        for index, (key, _) in enumerate(requests):
            progress(index / total, desc=f"Pose {index + 1}/{total}: {poses.POSES[key].label}")
            result = next(iterator)
            gallery.append((str(batch.folder / result["image"]), result["label"]))
            yield list(gallery), None, (
                f"Completed **{index + 1}/{total}** · seed **{used_seed}** · "
                f"{result['label']} in {result['total_seconds']:.1f}s"
            )
        # Exhaust the generator so its manifest records completion before creating the ZIP.
        next(iterator, None)
        archive = poses.zip_batch(batch)
        progress(1, desc="Poses ready")
        yield list(gallery), str(archive), (
            f"Completed **{total}/{total}** · seed **{used_seed}** · **{VRAM_PRESETS[vram_preset]}**\n\n"
            f"Saved to `outputs/poses/{batch.folder.name}`. ZIP includes images, prompts, settings and references."
        )
    except Exception as exc:
        log.exception("Pose generation failed")
        archive = None
        if batch is not None:
            try:
                archive = str(poses.zip_batch(batch))
            except Exception:
                log.exception("Could not package partial pose batch")
        yield list(gallery), archive, f"**Stopped after {len(gallery)}/{len(requests)} poses and expressions:** {exc}. Completed images remain saved."
    finally:
        model.set_cancel_event(None)
        if iterator is not None:
            iterator.close()


def poses_tab(vram_preset) -> None:
    gr.Markdown(
        "Upload a subject and choose new poses and expressions. Extra references should show the same subject. "
        "Each image uses the uploaded originals and the same seed. Prompts preserve identity and appearance "
        "while changing the pose and expression. Use the VRAM preset above; Small size is useful for trials."
    )
    references = gr.Gallery(
        label="Reference images (1–10, first image sets original output size)", columns=4,
        object_fit="contain", interactive=True, sources=["upload"], type="filepath",
        file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    )
    selected = gr.CheckboxGroup(
        choices=[(item.label, key) for key, item in poses.POSES.items()],
        value=list(poses.POSES), label="Poses and expressions to generate",
    )
    instructions = gr.Textbox(
        label="Additional instructions for every image (optional)", lines=2,
        placeholder="For example: keep the jacket's pattern and eye color",
    )
    with gr.Accordion("Pose and expression prompts", open=False):
        prompt_inputs = []
        for item in poses.POSES.values():
            prompt_inputs.append(gr.Textbox(label=item.label, value=item.prompt, lines=4))
            gr.Textbox(
                label=f"Saved negative prompt · {item.label} · CFG {item.true_cfg_scale:g}",
                value=item.negative_prompt, lines=2, interactive=False,
            )
        reset = gr.Button("Reset prompts")
        reset.click(lambda: [item.prompt for item in poses.POSES.values()], outputs=prompt_inputs, queue=False)
    reference_quality = gr.Dropdown(
        choices=[ORIGINAL_SIZE, *REFERENCE_QUALITY_PIXELS], value=ORIGINAL_SIZE,
        label="Reference image size",
    )
    aspect, quality, steps, seed, transparent, cfg, negative, _, _ = controls("poses", include_output=False)
    use_pose_settings = gr.Checkbox(
        value=True, label="Use saved per-preset negative prompts and CFG",
        info="Uses the saved values shown under Pose and expression prompts. "
             "Turn off to use the shared CFG and negative prompt controls above.",
    )
    button = gr.Button("Generate selected poses and expressions", variant="primary")
    gallery = gr.Gallery(
        label="Generated poses and expressions", columns=3, rows=2,
        object_fit="contain", interactive=False,
    )
    archive = gr.File(label="Download poses and expressions ZIP", interactive=False)
    status = gr.Markdown()
    button.click(
        run_poses,
        inputs=[references, selected, instructions, aspect, quality, steps, seed, transparent,
                cfg, negative, reference_quality, vram_preset, *prompt_inputs, use_pose_settings],
        outputs=[gallery, archive, status], concurrency_limit=1, concurrency_id="gpu",
        show_progress="minimal",
    )


def _pose_review_updates(items, message):
    updates = [items, message]
    for i in range(10):
        item = items[i] if i < len(items) else None
        valid = bool(item and item.get("extracted_prompt") and not item.get("error"))
        updates.extend([
            gr.update(visible=item is not None),
            gr.update(value=item["path"] if item else None),
            gr.update(value=valid, interactive=valid),
            gr.update(value=item.get("reviewed_prompt", "") if item else "", interactive=valid,
                      label=(item["filename"] + (" · " + item["error"] if item.get("error") else "")) if item else "Pose prompt"),
        ])
    return updates


def invalidate_pose_uploads(values, items):
    try:
        pose_references.assert_current(items, values)
        return [gr.skip()] * 43
    except (ValueError, OSError):
        return [*_pose_review_updates([], "Pose uploads changed. Click Analyze poses to create prompts."), None]


def analyze_pose_uploads(values, previous, *prompts):
    try:
        pose_references.assert_current(previous, values)
        items = pose_references.review(previous, prompts) if prompts else [dict(i) for i in previous]
    except (ValueError, OSError):
        try:
            items = pose_references.new_items(values)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
    pending = [i for i in items if not i.get("extracted_prompt") or i.get("error")]
    if not pending:
        yield _pose_review_updates(items, "All poses are analyzed. Review the prompts before generating.")
        return
    _cancel_event.clear()
    yield _pose_review_updates(items, "Analyzing poses. First use downloads about 4.3 GB; watch the console.")
    iterator = pose_analyzer.analyze(pending, _cancel_event)
    try:
        for result in iterator:
            items = [result if i["id"] == result["id"] else i for i in items]
            done = sum(bool(i.get("extracted_prompt")) and not i.get("error") for i in items)
            yield _pose_review_updates(items, f"Analyzed {done}/{len(items)}. Review prompts; failed items can be reanalyzed.")
    except Exception as exc:
        log.exception("Pose analysis stopped")
        yield _pose_review_updates(items, f"Analysis stopped: {exc}. Successful descriptions are retained.")
    finally:
        iterator.close()


def save_pose_library(values, items, name, parent, *prompts):
    try:
        pose_references.assert_current(items, values)
        reviewed = pose_references.review(items, prompts)
        identifier = pose_references.save_set(reviewed, name, parent)
        return gr.update(choices=pose_references.list_sets(), value=identifier), identifier, f"Saved new pose-set version: {identifier}"
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def load_pose_library(identifier):
    try:
        data = pose_references.load_set(identifier)
        items = data["items"]
        return [pose_references.paths([i["path"] for i in items]), data["name"], identifier,
                *_pose_review_updates(items, "Pose set loaded. Review prompts and select poses before generating.")]
    except (ValueError, OSError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc


def run_pose_references(subjects, values, items, aspect, quality, steps, seed,
                        transparent, cfg, negative, reference_quality, vram_preset, *review_values):
    try:
        pose_references.assert_current(items, values)
        selected = pose_references.review(items, review_values[10:], review_values[:10])
        requests = pose_references.prepare_requests(
            subjects, selected, aspect=aspect, quality=quality, steps=int(steps), seed=int(seed),
            transparent=transparent, true_cfg_scale=cfg, negative_prompt=negative,
            reference_quality=reference_quality, vram_preset=vram_preset,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    gallery, batch, iterator = [], None, None
    _cancel_event.clear()
    model.set_cancel_event(_cancel_event)
    try:
        pose_analyzer.reset()
        batch = pose_references.create_batch(requests)
        yield gallery, None, f"Starting {len(requests)} poses · seed {requests[0][1].seed}"
        iterator = pose_references.generate_batch(batch)
        for result in iterator:
            gallery.append((str(batch[0] / result["image"]), result["label"]))
            yield list(gallery), None, f"Completed {len(gallery)}/{len(requests)}"
        yield gallery, str(pose_references.zip_batch(batch)), f"Completed {len(gallery)}/{len(requests)} · saved in {batch[0]}"
    except Exception as exc:
        log.exception("Pose reference generation stopped")
        if iterator is not None:
            iterator.close()
        archive = str(pose_references.zip_batch(batch)) if batch else None
        yield gallery, archive, f"Stopped after {len(gallery)}/{len(requests)}: {exc}. Completed results are retained."
    finally:
        if iterator is not None:
            iterator.close()
        model.set_cancel_event(None)


def pose_references_tab(vram_preset):
    gr.Markdown("Upload subject images and separate pose examples showing one person or humanoid each. "
                "Analyze, review the descriptions, then apply selected poses. Only subject images go to Qwen Image Edit. "
                "Check hand contacts, left/right positions and expressions before generating. "
                "Text descriptions approximate poses; exact skeletal matching is not guaranteed.")
    gr.Markdown("Output aspect ratio defaults to **Pose image**, using each example's shape. "
                "Preserve input resolution uses that pose image's dimensions; other sizes retain its ratio. "
                "Choose Original image to use the first subject image's shape instead.")
    subjects = gr.Gallery(label="Subject images (1–10)", columns=4,
                          type="filepath", sources=["upload"], interactive=True,
                          file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"])
    examples = gr.Gallery(label="Pose examples (1–10, in upload order)", columns=4,
                          type="filepath", sources=["upload"], interactive=True,
                          file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"])
    state, parent = gr.State([]), gr.State(None)
    with gr.Row():
        analyze = gr.Button("Analyze poses", variant="primary")
        library = gr.Dropdown(choices=pose_references.list_sets(), label="Saved pose sets")
        refresh = gr.Button("Refresh library")
        load = gr.Button("Load pose set")
    name = gr.Textbox(label="Pose-set name", value="Pose set")
    save = gr.Button("Save pose set as new version")
    status = gr.Markdown()
    review_outputs, checks, prompts = [], [], []
    for _ in range(10):
        with gr.Row(visible=False) as row:
            thumbnail = gr.Image(label="Pose example", interactive=False, height=220)
            with gr.Column():
                check = gr.Checkbox(label="Generate this pose", value=False, interactive=False)
                prompt = gr.Textbox(label="Pose prompt", lines=5, interactive=False)
        review_outputs.extend([row, thumbnail, check, prompt])
        checks.append(check)
        prompts.append(prompt)
    outputs = [state, status, *review_outputs]
    examples.change(invalidate_pose_uploads, [examples, state], [*outputs, parent], queue=False)
    analyze.click(analyze_pose_uploads, [examples, state, *prompts], outputs, concurrency_id="gpu", concurrency_limit=1)
    save.click(save_pose_library, [examples, state, name, parent, *prompts], [library, parent, status])
    refresh.click(lambda: gr.update(choices=pose_references.list_sets()), outputs=library)
    load.click(load_pose_library, [library], [examples, name, parent, *outputs])
    reference_quality = gr.Dropdown(choices=[ORIGINAL_SIZE, *REFERENCE_QUALITY_PIXELS],
                                    value=ORIGINAL_SIZE, label="Subject reference image size")
    aspect, quality, steps, seed, transparent, cfg, negative, _, _ = controls("pose_references", include_output=False)
    generate = gr.Button("Generate selected poses", variant="primary")
    results = gr.Gallery(label="Generated poses", columns=3, interactive=False)
    archive = gr.File(label="Download pose batch ZIP", interactive=False)
    generation_status = gr.Markdown()
    generate.click(run_pose_references,
                   [subjects, examples, state, aspect, quality, steps, seed, transparent, cfg,
                    negative, reference_quality, vram_preset, *checks, *prompts],
                   [results, archive, generation_status], concurrency_id="gpu", concurrency_limit=1)


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
    if job.get("settings"):
        message += f"\n\nVRAM preset: **{VRAM_PRESETS[job['settings'].get('vram_preset', 'base')]}**"
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
            settings.get("vram_preset", "base"),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def run_video_job(job_id: str, aspect: str, quality: str, steps: float, seed: float,
                  prompt: str, transparent: bool, true_cfg_scale: float, negative_prompt: str,
                  legacy_long_edge: int | None, mode: str, progress=gr.Progress(),
                  vram_preset: str = "base"):
    if not job_id:
        raise gr.Error("Extract a video or load a saved job first.")
    try:
        require_job_mode(job_id, mode)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    event = threading.Event()
    _pause_flags[job_id] = event
    _cancel_event.clear()
    model.set_cancel_event(_cancel_event)
    updates: queue.Queue = queue.Queue()
    frame_acknowledged: queue.Queue = queue.Queue()

    def on_frame(done: int, total: int, remaining: float | None) -> None:
        updates.put(("frame", done, total, remaining))
        # Wait for the generator to send this preview. A separate signal keeps
        # the worker from consuming its own frame notification from the queue.
        frame_acknowledged.get()
        if event.is_set():
            raise PauseRequested()

    def process() -> None:
        try:
            result = process_job(
                job_id, legacy_long_edge, int(steps), int(seed), prompt,
                on_progress=on_frame, expected_mode=mode, aspect=aspect, quality=quality,
                transparent=transparent, true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt, vram_preset=vram_preset,
            )
            updates.put(("done", result))
        except Exception as exc:
            updates.put(("error", exc))

    worker = threading.Thread(target=process, daemon=True)
    video_path = None
    try:
        gallery = preview_frames(job_id, processed=True, include_latest=True)
        yield gallery, video_path, _video_status(load_job(job_id), "Processing video frames.")
        worker.start()
        while True:
            message = updates.get()
            if message[0] == "frame":
                _, done, total, remaining = message
                estimate = f" · about {remaining / 60:.1f} min remaining" if remaining is not None else ""
                progress(done / total, desc=f"Qwen frame {done}/{total}{estimate}")
                gallery = preview_frames(job_id, processed=True, include_latest=True)
                job = load_job(job_id)
                yield gallery, None, _video_status(job, f"Completed frame **{done}/{total}**.{estimate}")
                frame_acknowledged.put(None)
            elif message[0] == "done":
                video_path = str(message[1])
                job = load_job(job_id)
                yield preview_frames(job_id, processed=True, include_latest=True), video_path, _video_status(job, "Video ready to preview or download.")
                break
            else:
                exc = message[1]
                if isinstance(exc, PauseRequested):
                    job = load_job(job_id)
                    yield preview_frames(job_id, processed=True, include_latest=True), None, _video_status(job, "Paused after the current frame. Press Start / resume to continue.")
                else:
                    log.error("Video batch failed: %s", exc)
                    job = load_job(job_id)
                    yield preview_frames(job_id, processed=True, include_latest=True), None, _video_status(job, f"**Stopped:** {exc}. Fix the issue and press Start / resume.")
                break
    finally:
        _cancel_event.set()
        event.set()
        frame_acknowledged.put(None)
        if worker.ident is not None:
            worker.join()
        model.set_cancel_event(None)
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


def video_tab(mode: str, vram_preset) -> None:
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
         negative_prompt, size_note, legacy_long_edge, vram_preset],
    )
    aspect.change(video_size_note, [job_id, aspect, quality, legacy_long_edge], [size_note], queue=False)
    quality.change(video_size_note, [job_id, aspect, quality, legacy_long_edge], [size_note], queue=False)
    process_button.click(
        run_video_job, [job_id, aspect, quality, video_steps, video_seed, video_prompt,
                        transparent, cfg, negative_prompt, legacy_long_edge, mode_state, vram_preset],
        [result_gallery, completed_video, video_status],
        concurrency_limit=1, concurrency_id="gpu",
        show_progress="minimal",
    )
    pause_button.click(pause_video_job, [job_id], [video_status], queue=False)
    source_zip_button.click(lambda value, selected: download_frames(value, False, selected), [job_id, mode_state], [source_zip])
    result_zip_button.click(lambda value, selected: download_frames(value, True, selected), [job_id, mode_state], [result_zip])


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Qwen Image 2.1 Local") as demo:
        gr.Markdown(
            "# Qwen Image 2.1 Local\n"
            "Generate images, edit images, create perspectives, poses and expressions, upscale video, or experimentally edit a video. First use downloads about 33 GB of model files; "
            "the console shows download progress. Small size is quickest; Edit preserves input resolution by default."
        )
        with gr.Row():
            cancel_button = gr.Button("Cancel current job")
            reset_button = gr.Button("Reset model / clear VRAM")
        resource_status = gr.Markdown()
        cancel_button.click(cancel_current_job, outputs=[resource_status], queue=False)
        reset_button.click(reset_model, outputs=[resource_status], concurrency_limit=1, concurrency_id="gpu")
        vram_preset = gr.Dropdown(
            choices=[(label, value) for value, label in VRAM_PRESETS.items()],
            value="base", label="VRAM preset",
            info="Applied on the next run. Switching reloads the model. Size and references still affect VRAM.",
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
                        transparent, cfg, negative_prompt, gr.State(ORIGINAL_SIZE), vram_preset,
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
                reference_quality = gr.Dropdown(
                    choices=[ORIGINAL_SIZE, *REFERENCE_QUALITY_PIXELS],
                    value=ORIGINAL_SIZE,
                    label="Reference image size",
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
                        e_seed, e_transparent, e_cfg, e_negative_prompt, reference_quality, vram_preset,
                    ],
                    outputs=[e_output, e_status],
                    concurrency_limit=1,
                    concurrency_id="gpu",
                )
            with gr.Tab("Perspectives"):
                perspectives_tab(vram_preset)
            with gr.Tab("Poses & Expressions"):
                poses_tab(vram_preset)
            with gr.Tab("Pose References"):
                pose_references_tab(vram_preset)
            with gr.Tab("Video"):
                video_tab("upscale", vram_preset)
            with gr.Tab("Video Edit (experimental)"):
                video_tab("edit", vram_preset)
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
