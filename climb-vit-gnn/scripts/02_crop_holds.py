"""Crop hold patches from board images."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.crop_holds import build_placement_to_crop_mapping, crop_all_holds


def main() -> None:
    config = yaml.safe_load(open("configs/vit_gnn.yaml"))
    cfg = config["data"]

    print("Cropping holds from board images...")
    crops = crop_all_holds(
        db_path=cfg["db_path"],
        board_images_dir=cfg["board_images_dir"],
        crop_size=cfg["crop_size"],
        padding_factor=cfg["crop_padding_factor"],
        output_dir=cfg["hold_crops_dir"],
    )

    print(f"\nTotal crops: {len(crops)}")

    mapping = build_placement_to_crop_mapping(cfg["db_path"], crops)

    mapping_path = Path(cfg["hold_crops_dir"]) / "placement_to_crop.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    print(f"Saved {mapping_path}")


if __name__ == "__main__":
    main()
