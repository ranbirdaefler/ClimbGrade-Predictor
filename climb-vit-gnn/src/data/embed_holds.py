"""Compute and cache DINOv2 embeddings for hold crops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_dinov2(model_name: str, device: str) -> torch.nn.Module:
    """Load DINOv2 model from torch hub."""
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()
    return model


def get_transform() -> transforms.Compose:
    """DINOv2 preprocessing transform (ImageNet normalization)."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def embed_all_holds(
    crops_dir: str | Path,
    model_name: str,
    device: str,
    batch_size: int = 128,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """Compute DINOv2 CLS-token embeddings for all hold crops.

    Returns (embeddings_dict, hold_ids) where embeddings_dict maps
    hold_id -> numpy array of shape (embedding_dim,).
    """
    model = load_dinov2(model_name, device)
    transform = get_transform()

    crop_paths = sorted(Path(crops_dir).glob("hold_*.png"))
    if not crop_paths:
        crop_paths = sorted(Path(crops_dir).glob("hold_*.jpg"))
    print(f"Found {len(crop_paths)} hold crops")

    embeddings: dict[int, np.ndarray] = {}
    batch_tensors: list[torch.Tensor] = []
    batch_ids: list[int] = []

    for crop_path in tqdm(crop_paths, desc="Embedding holds"):
        hold_id = int(crop_path.stem.split("_")[1])
        img = Image.open(crop_path).convert("RGB")
        tensor = transform(img)
        batch_tensors.append(tensor)
        batch_ids.append(hold_id)

        if len(batch_tensors) == batch_size:
            batch = torch.stack(batch_tensors).to(device)
            embs = model(batch)
            for idx, hid in enumerate(batch_ids):
                embeddings[hid] = embs[idx].cpu().numpy()
            batch_tensors.clear()
            batch_ids.clear()

    if batch_tensors:
        batch = torch.stack(batch_tensors).to(device)
        embs = model(batch)
        for idx, hid in enumerate(batch_ids):
            embeddings[hid] = embs[idx].cpu().numpy()

    return embeddings, sorted(embeddings.keys())


def save_embeddings(
    embeddings: dict[int, np.ndarray],
    hold_ids: list[int],
    output_dir: str | Path,
) -> None:
    """Save embeddings as numpy matrix + hold ID index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_matrix = np.stack([embeddings[hid] for hid in hold_ids])
    np.save(output_dir / "embeddings.npy", emb_matrix)

    with open(output_dir / "hold_ids.json", "w") as f:
        json.dump(hold_ids, f)

    print(f"Saved {len(hold_ids)} embeddings of dim {emb_matrix.shape[1]}")
    print(f"  Shape: {emb_matrix.shape}")
    print(f"  File size: {emb_matrix.nbytes / 1e6:.1f} MB")


def load_embeddings(
    embeddings_dir: str | Path,
) -> tuple[np.ndarray, dict[int, int]]:
    """Load cached embeddings. Returns (matrix, hold_id_to_idx)."""
    embeddings_dir = Path(embeddings_dir)
    emb_matrix = np.load(embeddings_dir / "embeddings.npy")
    with open(embeddings_dir / "hold_ids.json") as f:
        hold_ids = json.load(f)
    hold_id_to_idx = {hid: i for i, hid in enumerate(hold_ids)}
    return emb_matrix, hold_id_to_idx
