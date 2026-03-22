"""Train the ViT-GNN model."""

import json
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
    config = yaml.safe_load(open("configs/vit_gnn.yaml"))
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Load embeddings
    print("Loading embeddings...")
    emb_matrix, crop_id_to_idx = load_embeddings(data_cfg["embeddings_dir"])
    print(f"  Embeddings: {emb_matrix.shape}")

    # Load placement -> crop mapping
    crops_dir = Path(data_cfg["hold_crops_dir"])
    with open(crops_dir / "placement_to_crop.json") as f:
        placement_to_crop_raw = json.load(f)
    placement_to_crop = {int(k): int(v) for k, v in placement_to_crop_raw.items()}
    print(f"  Placement -> crop entries: {len(placement_to_crop)}")

    # Load routes and splits
    print("\nLoading routes and splits...")
    all_routes = json.load(open(data_cfg["routes_path"]))
    splits = json.load(open(data_cfg["splits_path"]))

    train_uuids = set(splits["train"])
    val_uuids = set(splits["val"])
    test_uuids = set(splits["test"])

    train_routes = [r for r in all_routes if r["uuid"] in train_uuids]
    val_routes = [r for r in all_routes if r["uuid"] in val_uuids]
    test_routes = [r for r in all_routes if r["uuid"] in test_uuids]
    print(f"  Routes: train={len(train_routes)}, val={len(val_routes)}, test={len(test_routes)}")

    # Build datasets
    print("\nBuilding PyG datasets...")
    train_ds = KilterViTDataset(train_routes, emb_matrix, placement_to_crop, crop_id_to_idx, foot_drop_prob=0.8)
    val_ds = KilterViTDataset(val_routes, emb_matrix, placement_to_crop, crop_id_to_idx)
    test_ds = KilterViTDataset(test_routes, emb_matrix, placement_to_crop, crop_id_to_idx)
    print(f"  Graphs: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training graphs built. Check placement->crop mapping.")
        return

    sample = train_ds[0]
    print(f"  Sample: {sample.vit_emb.shape[0]} nodes, vit_emb={sample.vit_emb.shape}")

    nw = train_cfg["num_workers"]
    pw = nw > 0
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=nw, persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        num_workers=nw, persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"],
        num_workers=nw, persistent_workers=pw,
    )

    device = torch.device(config["vit"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    model = KilterViTGNN(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    print("\nTraining ViT-GNN...")
    history = train_vit_gnn(model, train_loader, val_loader, config, device)

    # Evaluate on test set
    from src.evaluate_vit import evaluate, plot_training_curves
    print("\nEvaluating on test set...")
    results = evaluate(model, test_loader, device)
    print(f"  Test MAE:  {results['mae']:.4f}")
    print(f"  Test RMSE: {results['rmse']:.4f}")
    print(f"  +-1 Accuracy: {results['within_1']:.1f}%")

    plot_training_curves(history)

    # Save results
    save_data = {"mae": results["mae"], "rmse": results["rmse"], "within_1": results["within_1"]}
    with open("results/vit_gnn_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("\n  Saved results/vit_gnn_results.json")


if __name__ == "__main__":
    main()
