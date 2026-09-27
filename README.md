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

Each result is saved as a PNG in `outputs/`, beside a JSON file with its prompt, effective prompt, settings, seed, and reference filenames. Input images are read from Gradio's temporary upload paths and are not copied into `outputs/`. Only one request runs at a time.

## Video frames

1. Open **Video**, upload a clip, choose output FPS (10 by default) and optional start/end times, then click **Extract frames**. Review the source frame samples and displayed frame count before starting Qwen. The chosen FPS controls both frame sampling and MP4 playback, keeping approximately the same duration. If it exceeds the source FPS, FFmpeg repeats frames; this feature does not create new motion.
2. Choose the output frame aspect ratio and size, Qwen steps (40 by default), seed, and preservation prompt. As in Edit, **Original image** and **Preserve input resolution** are the defaults. Other aspect ratios and Small, Standard, 1.5 MP, or 2K sizes are available. Transparency, CFG scale (1.0 by default), and a negative prompt use the same controls as Edit. Transparent PNG frames can be downloaded; the completed H.264 MP4 has no alpha channel. Internally, Qwen uses dimensions rounded up to its required 32-pixel grid, then each result is resized to the selected output size. Click **Start / resume Qwen batch**. Each frame is processed independently with the same prompt and seed. A large batch can take many hours.
3. Use **Pause after current frame** to stop after the active inference finishes. The Job ID and completed frames stay in `outputs/video_jobs/<job-id>/`. To continue after restarting the app, enter the Job ID, click **Load saved job**, then **Start / resume**. Settings are locked after the first processed frame. Failed jobs can also resume after the issue is fixed. Older jobs retain their saved long-edge output dimensions when resumed.

When all frames are ready, the app saves an H.264 MP4 with source audio when present. The Video tab previews representative source and Qwen frames, plays the result, and offers ZIP downloads of either frame set. All source files, frames, and output videos remain in the job folder until you delete that folder. Qwen may reinterpret details or create flicker between independently processed frames, even with the preservation prompt.

For performance debugging, each processed frame adds a timing record to `outputs/video_jobs/<job-id>/timings.jsonl` and a summary to the console. The record includes the time between frames, pipeline loading, input preparation, time to the first model step, total model execution, frame preparation, PNG saving, and total frame time. The first-step time includes prompt encoding and the first denoising step.

## Experimental video edit

Open **Video Edit (experimental)**, upload a clip, and extract frames as above. Enter one change to apply to every frame, such as “Change the red car to blue.” The app asks Qwen to preserve everything else, but each frame is edited independently, so the change may vary or flicker. Try a short clip at a low FPS first. This tab has its own saved jobs, pause/resume flow, edited frame ZIP, and completed MP4 with source audio. Its results are stored in `outputs/video_jobs/<job-id>/edited_frames/` and `edited.mp4`.

The model is distributed under the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE). Read its terms before using or distributing outputs beyond personal testing.
