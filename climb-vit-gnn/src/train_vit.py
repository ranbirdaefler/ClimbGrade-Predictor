"""Training loop for the ViT-GNN model."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_vit_gnn(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    save_dir: str = "results",
) -> dict:
    """Train the ViT-GNN model with early stopping on val MAE."""
    cfg = config["training"]
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_mae": []}

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

        avg_loss = total_loss / n_samples
        history["train_loss"].append(avg_loss)

        val_mae = _eval_mae(model, val_loader, device)
        history["val_mae"].append(val_mae)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d} | loss {avg_loss:.4f} | val MAE {val_mae:.4f} | lr {lr:.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch} (best val MAE: {best_val_mae:.4f})")
                break

    model.load_state_dict(best_state)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(best_state, Path(save_dir) / "vit_gnn_best.pt")
    return history


@torch.no_grad()
def _eval_mae(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_ae = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        total_ae += (pred - batch.y.squeeze()).abs().sum().item()
        n += batch.num_graphs
    return total_ae / n
