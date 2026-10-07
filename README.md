# Qwen Image 2.1 Local

A local Windows web interface for the official [Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) model. It offers text-to-image generation, image editing with up to ten reference images, video frame upscaling, and experimental video editing.

For a GPU Pod, see [Runpod quick start](RUNPOD.md).

## Requirements

- NVIDIA GPU with 16 GB VRAM (tested target: RTX 5070 Ti), current driver, and sufficient system RAM for CPU offload.
- Around 45 GB of free disk space for the model, Python environment, and cache, plus space for each video job's source, extracted PNGs, results, and optional ZIP downloads.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git installed on Windows.
- Internet access for the first setup and model download.

## Install and run

Open PowerShell in this folder:

```powershell
.\setup.cmd
.\launch.cmd
```

The `.cmd` launchers run the PowerShell scripts with an execution-policy override limited to that process. They do not change your system or user policy. `setup.cmd` creates a local Python 3.12 environment, installs CUDA PyTorch and the pinned Diffusers source needed for `QwenImage21Pipeline`, then checks CUDA and the pipeline import. `launch.cmd` opens the local interface at `http://127.0.0.1:7860`. No account or network sharing is enabled.

The first Generate, Edit, or Video batch downloads the official BF16 checkpoint (about 33 GB) to `models/Qwen-Image-2.1/`. Watch the PowerShell window for download and model loading progress. The fixed checkpoint revision is reused on later runs. Files are downloaded directly to this folder because Windows may restrict symbolic links used by the default Hugging Face cache. The model loads once per app session and uses CPU offload to fit the 16 GB GPU. A request can take several minutes.

Use **Small (512×512)** for quicker tests. Other aspect ratios use roughly the same pixel count at the selected shape. **Standard (~1 MP)** remains the Generate default. **1.5 MP** offers more detail with higher memory use. **2K** is experimental on 16 GB and may exhaust GPU memory, particularly with several references. A seed of `-1` chooses a random seed; the chosen value is shown with the result. The transparency checkbox adds the model's recommended RGBA wording to the prompt. Actual alpha content depends on model output.

The Generate and Edit tabs include a collapsed **Advanced guidance** section. Qwen Image 2.1 is designed to run without classifier-free guidance, so **CFG scale** defaults to `1.0`. Set it above `1.0` to strengthen prompt adherence. A blank negative prompt uses unconditional guidance; enter a negative prompt to describe content or qualities to avoid. Negative prompts have no effect at CFG `1.0`, and enabling CFG increases processing time because the model evaluates both positive and negative guidance.

In **Edit**, **Original image** and **Preserve input resolution** are the defaults. The saved result uses the first uploaded reference's exact width and height. Qwen runs at the nearest larger 16-pixel grid size when necessary, then the result is resized to the original dimensions. Choose Small, Standard, 1.5 MP, or 2K to use a different pixel count while retaining the reference's shape. Choosing a named aspect ratio switches the size to Standard; selecting Preserve input resolution switches the aspect ratio back to Original image. Large input images may exhaust GPU memory. **Generate** still defaults to 1:1 at Standard size.

Edit also has a **Reference image size** setting, which defaults to **Preserve input resolution**. Extra small (256×256), Small (512×512), Standard (~1 MP), 1.5 MP, and 2K set a maximum pixel area for each reference image. Larger images are resized proportionally before inference; smaller images remain unchanged. This setting does not change the output size controls.

Each result is saved as a PNG in `outputs/`, beside a JSON file with its prompt, effective prompt, settings, seed, and reference filenames. Input images are read from Gradio's temporary upload paths and are not copied into `outputs/`. Only one request runs at a time.

## VRAM presets

The shared **VRAM preset** dropdown applies to all six tabs. **Base — BF16** is the default and retains the original loading behavior. **Medium VRAM — 8-bit** quantizes the transformer and text encoder to 8-bit. **Low VRAM — 4-bit** uses NF4 quantization with BF16 computation. All presets use CPU offload and the same downloaded checkpoint; no extra checkpoint download is needed. Quantization happens during loading and may change output quality or speed.

After updating, rerun `setup.cmd` (or `setup_runpod.sh` on Runpod) to install the pinned bitsandbytes dependency. Changing the dropdown takes effect on the next generation or batch and reloads the pipeline; it does not interrupt an active request. Video jobs save and lock their preset with other settings, and loading a job restores its preset. Older saved jobs use Base.

Size, guidance, and reference images still affect peak VRAM. These options do not establish support for GPUs below the current 16 GB requirement. The console reports the selected preset and measured memory use.

The loader also moves bitsandbytes' internal 8-bit state with CPU offload, avoiding extra GPU copies. Compare output quality before running a long batch.
Medium retains the small conditioning and input/output projections in BF16 to protect generation quality; the transformer blocks and text encoder use 8-bit quantization.

## Perspectives

The **Perspectives** tab creates selected views from 1–10 reference images of the same subject: **3/4 body (head to mid-thigh)**, **1/2 body (head and torso)**, **full body (head to feet)**, **bird's-eye**, **worm's-eye**, and **extreme face close-up**. These prompts generalize the robot experiment while preserving the uploaded subject's appearance, background, lighting and style. Open **Perspective prompts** to adjust individual prompts or add shared instructions. Body crops and face close-ups are intended for people and characters; adapt the prompts for other subjects. Inspect the resulting crop and camera angle, as adherence can vary.

Choose any subset and click **Generate selected perspectives**. Results appear in the gallery as each image finishes. The tab uses the shared VRAM preset, defaults to 40 steps and seed 42, and preserves the first reference's output resolution. A random seed (`-1`) is chosen once and reused for every selected view. Each view uses the uploaded originals, rather than a previous generated image.

Batches are saved in `outputs/perspectives/<batch-id>/`, including retained references, PNGs, prompt/settings JSON files, timings and `batch.json`. The downloadable ZIP contains the whole batch. If a view fails, processing stops and the completed images remain available with a partial ZIP.

## Poses & Expressions

The **Poses & Expressions** tab generates **Wall lean**, **Victory cheer**, **Flirty smile** (index finger touching the lower lip), **Confident hero**, **Curious thinker**, and **Surprised reaction** from 1–10 reference images of the same subject. All six are initially selected, with results displayed in a three-column, two-row grid. The prompts preserve identity, facial structure, clothing or exterior design, lighting and style while changing pose and expression. Wall lean can add a plain wall; the other presets retain the reference setting. Wall lean, victory cheer and confident hero show the whole body; flirty smile, curious thinker and surprised reaction are waist-up portraits. These presets are intended for people and humanoid characters; adapt the editable prompts for other subjects.

Choose any subset and click **Generate selected poses and expressions**. Open **Pose and expression prompts** to edit individual prompts or reset them to the tuned defaults. Shared additional instructions apply to every selected image. The tab uses the shared VRAM preset and GPU queue, defaults to 40 steps and seed 42, and preserves the first reference's output resolution. A seed of `-1` chooses one random seed for the whole batch. Each image uses the uploaded originals rather than another generated result. Size, aspect ratio, reference size, transparency and advanced guidance work as in Perspectives.

Results appear as each image finishes and are saved in `outputs/poses/<batch-id>/`. The downloadable ZIP includes retained references, PNGs, prompt/settings metadata, timings and `batch.json`. Every selected prompt is validated before generation. If generation fails, processing stops and completed images remain available with a partial ZIP.

The default prompts are tuned during development using separate prompting and visual-review agents against the retained robot reference. Low VRAM trials use 512×512, 40 steps and seed 42. Reviews equally score action/expression adherence, identity fidelity, anatomy/mechanical coherence and image quality. A passing trial averages at least 8/10 with adherence and fidelity each at least 8. Each preset has a ten-output cap; if none passes, the highest-scoring trial is retained. Selected prompts are rendered once in Medium VRAM at the original reference resolution. Trial artifacts, reviews, timings, comparisons and any missed targets or final regressions are retained in `outputs/poses_six_robot_20261007/report.md` (the original four-preset experiment remains in `outputs/poses_robot_20261007/`). These reviews are development checks for that reference; inspect results for other subjects.

## Video frames

1. Open **Video**, upload a clip, choose output FPS (10 by default) and optional start/end times, then click **Extract frames**. Review the source frame samples and displayed frame count before starting Qwen. The chosen FPS controls both frame sampling and MP4 playback, keeping approximately the same duration. If it exceeds the source FPS, FFmpeg repeats frames; this feature does not create new motion.
2. Choose the output frame aspect ratio and size, Qwen steps (40 by default), seed, and preservation prompt. As in Edit, **Original image** and **Preserve input resolution** are the defaults. Other aspect ratios and Small, Standard, 1.5 MP, or 2K sizes are available. Transparency, CFG scale (1.0 by default), and a negative prompt use the same controls as Edit. Transparent PNG frames can be downloaded; the completed H.264 MP4 has no alpha channel. Internally, Qwen uses dimensions rounded up to its required 32-pixel grid, then each result is resized to the selected output size. Click **Start / resume Qwen batch**. Each frame is processed independently with the same prompt and seed. A large batch can take many hours.
3. Use **Pause after current frame** to stop after the active inference finishes. The Job ID and completed frames stay in `outputs/video_jobs/<job-id>/`. To continue after restarting the app, enter the Job ID, click **Load saved job**, then **Start / resume**. Settings are locked after the first processed frame. Failed jobs can also resume after the issue is fixed. Older jobs retain their saved long-edge output dimensions when resumed.

When all frames are ready, the app saves an H.264 MP4 with source audio when present. The Video tab previews representative source and Qwen frames, plays the result, and offers ZIP downloads of either frame set. All source files, frames, and output videos remain in the job folder until you delete that folder. Qwen may reinterpret details or create flicker between independently processed frames, even with the preservation prompt.

For performance debugging, each processed frame adds a timing record to `outputs/video_jobs/<job-id>/timings.jsonl` and a summary to the console. The record includes the time between frames, pipeline loading, input preparation, time to the first model step, total model execution, frame preparation, PNG saving, and total frame time. The first-step time includes prompt encoding and the first denoising step.

## Experimental video edit

Open **Video Edit (experimental)**, upload a clip, and extract frames as above. Enter one change to apply to every frame, such as “Change the red car to blue.” The app asks Qwen to preserve everything else, but each frame is edited independently, so the change may vary or flicker. Try a short clip at a low FPS first. This tab has its own saved jobs, pause/resume flow, edited frame ZIP, and completed MP4 with source audio. Its results are stored in `outputs/video_jobs/<job-id>/edited_frames/` and `edited.mp4`.

The model is distributed under the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE). Read its terms before using or distributing outputs beyond personal testing.
