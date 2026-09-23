# Runpod quick start

1. Deploy a GPU Pod with at least 16 GB VRAM and enough RAM for CPU offload. Choose a PyTorch or CUDA image with a driver compatible with CUDA 12.8. Mount persistent storage at `/workspace`; allow at least 45 GB for the model and environment, plus room for video jobs. In the Pod template's **Expose HTTP Ports** field, add `7860` (HTTP). Port 22 is only needed if you plan to use SSH.
2. Open a Pod terminal and run:

   ```bash
   cd /workspace
   git clone https://github.com/pjStyle/qwen_image_2.1_testing.git
   cd qwen_image_2.1_testing
   bash setup_runpod.sh
   read -rsp 'App password: ' QWEN_AUTH_PASSWORD; echo
   export QWEN_AUTH_PASSWORD
   bash launch_runpod.sh
   ```

3. In Runpod, open the Pod's **Connect** page and select the HTTP service on port `7860`, or visit `https://<pod-id>-7860.proxy.runpod.net`. Sign in as `qwen` with the password you set. Keep the terminal running while you use the app.

The app listens on `0.0.0.0:7860` inside the Pod so Runpod's HTTP proxy can reach it. The password is required because the proxy URL is externally accessible. To use another exposed HTTP port, set `QWEN_PORT` to that number before launching and expose the same port in Runpod.

Keep this repository under `/workspace` so `models/`, `outputs/`, and caches survive Pod restarts. The first generation downloads about 33 GB. Save anything important from `outputs/` before deleting the Pod or its storage. For later starts, run only `bash launch_runpod.sh` after setting the password in that terminal.

Runpod references: [Pod ports and proxy URL](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods), [persistent storage](https://docs.runpod.io/pods/troubleshooting/zero-gpus).
