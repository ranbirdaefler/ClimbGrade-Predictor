"""Evaluate ViT-GNN and compare with discrete GNN results."""

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
from src.evaluate_vit import evaluate, plot_scatter, print_comparison
from src.models.gnn_vit import KilterViTGNN


def main() -> None:
    config = yaml.safe_load(open("configs/vit_gnn.yaml"))
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Load embeddings + mapping
    emb_matrix, crop_id_to_idx = load_embeddings(data_cfg["embeddings_dir"])
    with open(Path(data_cfg["hold_crops_dir"]) / "placement_to_crop.json") as f:
        placement_to_crop = {int(k): int(v) for k, v in json.load(f).items()}

    # Load test routes
    all_routes = json.load(open(data_cfg["routes_path"]))
    splits = json.load(open(data_cfg["splits_path"]))
    test_uuids = set(splits["test"])
    test_routes = [r for r in all_routes if r["uuid"] in test_uuids]

    test_ds = KilterViTDataset(test_routes, emb_matrix, placement_to_crop, crop_id_to_idx)
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"])

    device = torch.device(config["vit"]["device"] if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = KilterViTGNN(config).to(device)
    checkpoint = torch.load("results/vit_gnn_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    print("Evaluating ViT-GNN on test set...")
    vit_results = evaluate(model, test_loader, device)

    # Load discrete GNN results for comparison
    all_results = {"GNN (ViT frozen)": vit_results}

    discrete_path = Path("../climb-pred/results/gnn_results.json")
    if discrete_path.exists():
        discrete = json.load(open(discrete_path))
        all_results["GNN (discrete)"] = discrete

    baseline_path = Path("../climb-pred/results/baseline_results.json")
    if baseline_path.exists():
        baselines = json.load(open(baseline_path))
        for name, res in baselines.items():
            all_results[name] = res

    print_comparison(all_results)

    # Only plot scatter for models that have predictions
    plot_results = {k: v for k, v in all_results.items() if "predictions" in v}
    if plot_results:
        plot_scatter(plot_results)


if __name__ == "__main__":
    main()
