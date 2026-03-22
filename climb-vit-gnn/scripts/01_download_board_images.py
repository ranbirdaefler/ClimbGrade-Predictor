"""Download Kilter Board images via boardlib."""

import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    config = yaml.safe_load(open("configs/vit_gnn.yaml"))
    out_dir = Path(config["data"]["board_images_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = config["data"]["db_path"]
    print(f"Downloading board images from {db_path} -> {out_dir}")
    print("(This requires your Kilter Board credentials)")

    result = subprocess.run(
        [sys.executable, "-m", "boardlib", "images", "kilter", db_path, str(out_dir)],
        text=True,
    )

    if result.returncode != 0:
        print(f"boardlib exited with code {result.returncode}")
        print("If this fails, you can manually place board images in", out_dir)
        return

    images = list(out_dir.glob("*"))
    print(f"\nDownloaded {len(images)} files to {out_dir}")
    for img in images[:10]:
        print(f"  {img.name}")
    if len(images) > 10:
        print(f"  ... and {len(images) - 10} more")


if __name__ == "__main__":
    main()
