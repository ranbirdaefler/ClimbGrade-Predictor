"""Compute DINOv2 embeddings for all hold crops."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.embed_holds import embed_all_holds, save_embeddings


def main() -> None:
    config = yaml.safe_load(open("configs/vit_gnn.yaml"))
    vit_cfg = config["vit"]
    data_cfg = config["data"]

    crops_dir = data_cfg["hold_crops_dir"]
    out_dir = data_cfg["embeddings_dir"]

    print(f"Model: {vit_cfg['model_name']}")
    print(f"Device: {vit_cfg['device']}")
    print(f"Batch size: {vit_cfg['batch_size']}")

    embeddings, hold_ids = embed_all_holds(
        crops_dir,
        model_name=vit_cfg["model_name"],
        device=vit_cfg["device"],
        batch_size=vit_cfg["batch_size"],
    )

    save_embeddings(embeddings, hold_ids, out_dir)


if __name__ == "__main__":
    main()
