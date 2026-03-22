"""Retrain ViT-GNN on the full dataset for deployment."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset_vit import KilterViTDataset
from src.data.embed_holds import load_embeddings
from src.models.gnn_vit import KilterViTGNN
from src.train_vit import train_vit_gnn


def main() -> None:
    config = yaml.safe_load(open("configs/retrain_full.yaml"))
    data_cfg = config["data"]
    train_cfg = config["training"]

    print("Loading embeddings...")
    emb_matrix, crop_id_to_idx = load_embeddings(data_cfg["embeddings_dir"])
    print(f"  Embeddings: {emb_matrix.shape}")

    crops_dir = Path(data_cfg["hold_crops_dir"])
    with open(crops_dir / "placement_to_crop.json") as f:
        placement_to_crop = {int(k): int(v) for k, v in json.load(f).items()}
    print(f"  Placement -> crop entries: {len(placement_to_crop)}")

    print("\nLoading routes...")
    all_routes = json.load(open(data_cfg["routes_path"]))
    print(f"  Total routes: {len(all_routes)}")

    # 95/5 split by UUID
    uuids = list({r["uuid"] for r in all_routes})
    random.seed(data_cfg["random_seed"])
    random.shuffle(uuids)
    val_count = int(len(uuids) * data_cfg["val_size"])
    val_uuids = set(uuids[:val_count])
    train_uuids = set(uuids[val_count:])

    train_routes = [r for r in all_routes if r["uuid"] in train_uuids]
    val_routes = [r for r in all_routes if r["uuid"] in val_uuids]
    print(f"  Train: {len(train_routes)}  Val: {len(val_routes)}")

    print("\nBuilding PyG datasets...")
    train_ds = KilterViTDataset(train_routes, emb_matrix, placement_to_crop, crop_id_to_idx, foot_drop_prob=0.8)
    val_ds = KilterViTDataset(val_routes, emb_matrix, placement_to_crop, crop_id_to_idx)
    print(f"  Graphs: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"])

    device = torch.device(config["vit"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    model = KilterViTGNN(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    print("\nTraining on full dataset...")
    history = train_vit_gnn(model, train_loader, val_loader, config, device, save_dir="results")

    # Save deployment checkpoint
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    best_val_mae = min(history["val_mae"])
    epochs_trained = len(history["train_loss"])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "crop_id_to_idx": crop_id_to_idx,
            "placement_to_crop": placement_to_crop,
            "training_stats": {
                "train_routes": len(train_ds),
                "val_routes": len(val_ds),
                "best_val_mae": best_val_mae,
                "epochs_trained": epochs_trained,
            },
        },
        models_dir / "deployment_model.pt",
    )
    print(f"\nSaved deployment model to {models_dir / 'deployment_model.pt'}")
    print(f"  Best val MAE: {best_val_mae:.4f}")
    print(f"  Epochs trained: {epochs_trained}")


if __name__ == "__main__":
    main()
