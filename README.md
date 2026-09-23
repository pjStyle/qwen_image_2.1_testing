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

Use **Small (512×512)** for quicker tests. Other aspect ratios use roughly the same pixel count at the selected shape. **Standard (~1 MP)** remains the default. **1.5 MP** offers more detail with higher memory use. **2K** is experimental on 16 GB and may exhaust GPU memory, particularly with several references. A seed of `-1` chooses a random seed; the chosen value is shown with the result. The transparency checkbox adds the model's recommended RGBA wording to the prompt. Actual alpha content depends on model output.

Each result is saved as a PNG in `outputs/`, beside a JSON file with its prompt, effective prompt, settings, seed, and reference filenames. Input images are read from Gradio's temporary upload paths and are not copied into `outputs/`. Only one request runs at a time.

## Video frames

1. Open **Video**, upload a clip, choose output FPS (10 by default) and optional start/end times, then click **Extract frames**. Review the source frame samples and displayed frame count before starting Qwen. The chosen FPS controls both frame sampling and MP4 playback, keeping approximately the same duration. If it exceeds the source FPS, FFmpeg repeats frames; this feature does not create new motion.
2. Choose a target long edge from 256 to 2048 pixels, Qwen steps (40 by default), seed, and preservation prompt. The output keeps the source aspect ratio; internally, Qwen uses dimensions rounded up to its required 32-pixel grid, then each result is resized to the displayed output size. A target below the source size downsizes the result. Click **Start / resume Qwen batch**. Each frame is processed independently with the same prompt and seed. A large batch can take many hours.
3. Use **Pause after current frame** to stop after the active inference finishes. The Job ID and completed frames stay in `outputs/video_jobs/<job-id>/`. To continue after restarting the app, enter the Job ID, click **Load saved job**, then **Start / resume**. Settings are locked after the first processed frame. Failed jobs can also resume after the issue is fixed.

When all frames are ready, the app saves an H.264 MP4 with source audio when present. The Video tab previews representative source and Qwen frames, plays the result, and offers ZIP downloads of either frame set. All source files, frames, and output videos remain in the job folder until you delete that folder. Qwen may reinterpret details or create flicker between independently processed frames, even with the preservation prompt.

## Experimental video edit

Open **Video Edit (experimental)**, upload a clip, and extract frames as above. Enter one change to apply to every frame, such as “Change the red car to blue.” The app asks Qwen to preserve everything else, but each frame is edited independently, so the change may vary or flicker. Try a short clip at a low FPS first. This tab has its own saved jobs, pause/resume flow, edited frame ZIP, and completed MP4 with source audio. Its results are stored in `outputs/video_jobs/<job-id>/edited_frames/` and `edited.mp4`.

The model is distributed under the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE). Read its terms before using or distributing outputs beyond personal testing.
