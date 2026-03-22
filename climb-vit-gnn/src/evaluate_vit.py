"""Evaluation and comparison for the ViT-GNN model."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    """Evaluate model on a DataLoader, return metrics + predictions."""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        preds.append(pred.cpu().numpy())
        trues.append(batch.y.squeeze().cpu().numpy())

    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    errors = pred - true
    ae = np.abs(errors)

    per_grade: dict[int, float] = {}
    for p, t in zip(pred, true):
        b = int(np.floor(t))
        per_grade.setdefault(b, []).append(abs(p - t))
    per_grade_mae = {b: float(np.mean(v)) for b, v in sorted(per_grade.items())}

    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "within_1": float((ae <= 1.0).mean() * 100),
        "predictions": pred,
        "targets": true,
        "per_grade_mae": per_grade_mae,
    }


def print_comparison(results: dict[str, dict]) -> None:
    """Print comparison table across models."""
    print("\n" + "=" * 55)
    print(f"  {'Model':<20s} | {'MAE':>6s} | {'RMSE':>6s} | {'+-1 Acc':>7s}")
    print("-" * 55)
    for name, res in results.items():
        mae = res["mae"] if isinstance(res["mae"], float) else res["mae"]
        rmse = res["rmse"] if isinstance(res["rmse"], float) else res["rmse"]
        w1 = res["within_1"] if isinstance(res["within_1"], float) else res["within_1"]
        print(f"  {name:<20s} | {mae:6.3f} | {rmse:6.3f} | {w1:5.1f}%")
    print("=" * 55)


def plot_training_curves(history: dict, save_dir: str = "results") -> None:
    """Plot train loss and val MAE curves."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss (MSE)")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")

    axes[1].plot(epochs, history["val_mae"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val MAE")
    axes[1].set_title("Validation MAE")

    plt.tight_layout()
    plt.savefig(save_path / "vit_gnn_training_curves.png", dpi=150)
    plt.close()
    print(f"  Saved {save_path / 'vit_gnn_training_curves.png'}")


def plot_scatter(results: dict[str, dict], save_dir: str = "results") -> None:
    """Scatter plot of predicted vs true grades."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for i, (name, res) in enumerate(results.items()):
        if "predictions" not in res:
            continue
        ax = axes[0][i]
        ax.scatter(res["targets"], res["predictions"], alpha=0.1, s=4)
        lims = [
            min(res["targets"].min(), res["predictions"].min()) - 1,
            max(res["targets"].max(), res["predictions"].max()) + 1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True Grade")
        ax.set_ylabel("Predicted Grade")
        ax.set_title(f"{name}\nMAE={res['mae']:.3f}")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path / "vit_gnn_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved {save_path / 'vit_gnn_scatter.png'}")
