def main() -> None:
    import torch
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch installation.")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 14 * 1024**3:
        raise RuntimeError("This setup expects a GPU with at least 16 GB of VRAM.")
    print(f"CUDA ready: {device.name}, {device.total_memory / 1024**3:.1f} GiB VRAM")
    print(f"PyTorch {torch.__version__}; Diffusers QwenImage21Pipeline available")


if __name__ == "__main__":
    main()
