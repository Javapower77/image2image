from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download, snapshot_download

from photo_edit_studio.config import settings
from photo_edit_studio.models import MODEL_SPECS
from photo_edit_studio.models.qwen_aio import is_safetensors_file

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
QWEN_AIO_REPO = "Phr00t/Qwen-Image-Edit-Rapid-AIO"
QWEN_AIO_REMOTE_FILE = "v23/Qwen-Rapid-AIO-SFW-v23.safetensors"
QWEN_AIO_LOCAL_NAME = "Qwen-Rapid-AIO.safetensors"


def _download_qwen_aio(local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / QWEN_AIO_LOCAL_NAME
    if is_safetensors_file(target):
        print(f"Already present: {target}")
        return
    if target.exists():
        target.unlink()
    print(f"Downloading {QWEN_AIO_REPO}/{QWEN_AIO_REMOTE_FILE} -> {target}")
    downloaded = Path(
        hf_hub_download(
            repo_id=QWEN_AIO_REPO,
            filename=QWEN_AIO_REMOTE_FILE,
            local_dir=local_dir,
        )
    )
    if downloaded != target:
        downloaded.replace(target)
    nested = local_dir / "v23"
    if nested.is_dir() and not any(nested.iterdir()):
        nested.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model snapshots into the local project.")
    parser.add_argument(
        "models", nargs="*", choices=list(MODEL_SPECS), help="Model keys; defaults to all"
    )
    parser.add_argument("--restorers", action="store_true", help="Download GFPGAN checkpoint")
    args = parser.parse_args()
    selected = args.models or list(MODEL_SPECS)
    for key in selected:
        spec = MODEL_SPECS[key]
        if spec.loader == "qwen_aio":
            _download_qwen_aio(spec.local_path)
            continue
        print(f"Downloading {spec.repo_id} -> {spec.local_path}")
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=spec.local_path,
            local_dir_use_symlinks=False,
        )
    if args.restorers:
        target = settings.model_dir / "restorers" / "GFPGANv1.4.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading GFPGAN -> {target}")
            urlretrieve(GFPGAN_URL, target)
    print("Downloads complete. Set HF_HUB_OFFLINE=1 for strictly offline launches.")


if __name__ == "__main__":
    main()
