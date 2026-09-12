# H100 Linux setup

## 1. Verify the host

Confirm `python3.11 --version` and `nvidia-smi` work. The setup script installs the CUDA 12.8 PyTorch wheel; the installed NVIDIA driver must support it. Do not install a second CUDA toolkit unless another application requires one.

## 2. Install

Run `bash scripts/setup.sh`. This creates `.venv`, installs PyTorch, installs a current Diffusers checkout, and installs development tools. To add GFPGAN, activate the environment and run `pip install -e '.[restore]'`.

## 3. Storage and authentication

By default, snapshots live in `models/repos`, Hugging Face cache data in `models/huggingface`, LoRAs in `models/loras`, restorer weights in `models/restorers`, and results in `outputs`. These paths are ignored by Git.

For gated repositories, accept the repository terms in a browser and run `huggingface-cli login` on the VM. FLUX Klein 9B has non-commercial terms. Never commit the token or put it in shell history; an environment-provided `HF_TOKEN` is also accepted by Hugging Face Hub.

## 4. Download

`python scripts/download_models.py flux-klein-4b` downloads one model. Multiple model keys may be listed. Add `--restorers` to fetch GFPGAN v1.4. Downloads can be large and resumable through Hugging Face Hub.

## 5. Network exposure

The default bind address is loopback. Prefer SSH port forwarding. If binding to `0.0.0.0`, set `PHOTO_EDIT_AUTH_USER` and `PHOTO_EDIT_AUTH_PASSWORD`, restrict the Azure NSG source range, and terminate TLS at a trusted reverse proxy. Never use Gradio share links for private photos.

## 6. Offline mode

After every required snapshot has downloaded, launch with `HF_HUB_OFFLINE=1`. Generation itself never calls a hosted inference API. FireRed’s optional hosted recaption agent is deliberately not integrated.

## Configuration

Copy `.env.example` to `.env`. Key settings include host, port, local directories, queue concurrency, and optional basic authentication. Keep concurrency at one to avoid simultaneous model loads and GPU OOM.
