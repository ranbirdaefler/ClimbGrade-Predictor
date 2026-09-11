"""Build a PyG graph from user-selected holds for GNN inference."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch_geometric.data import Data


def build_graph(
    hold_data: list[dict],
    wall_angle: float,
    image_shape: tuple[int, int],
) -> Data:
    """Construct a single PyG Data object for one route.

    Args:
        hold_data: list of dicts each with keys
            ``embedding`` (Tensor [768]), ``tap_x``, ``tap_y``, ``role``.
        wall_angle: wall angle in degrees.
        image_shape: (H, W) of the uploaded image.
    """
    n = len(hold_data)
    img_h, img_w = image_shape[:2]

    vit_embs = torch.stack([h["embedding"] for h in hold_data])

    # UI roles -> GNN role indices (trained on Kilter Board data)
    # "hand" and "volume" both map to the generic middle-hold index
    role_map = {"start": 0, "hand": 1, "middle": 1, "finish": 2, "foot": 3, "foot_only": 3, "volume": 1}
    role_idx = torch.tensor(
        [role_map.get(h["role"], 1) for h in hold_data], dtype=torch.long
    )

    # y is flipped: 0 = bottom of wall, 1 = top
    positions = torch.tensor(
        [[h["tap_x"] / img_w, 1.0 - h["tap_y"] / img_h] for h in hold_data],
        dtype=torch.float,
    )

    wall_angle_norm = torch.tensor([wall_angle / 70.0], dtype=torch.float)

    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)

    edge_attrs = []
    for s, d in zip(src, dst):
        dx = positions[d][0].item() - positions[s][0].item()
        dy = positions[d][1].item() - positions[s][1].item()
        dist = math.sqrt(dx * dx + dy * dy)
        ang = math.atan2(dy, dx)
        edge_attrs.append([dx, dy, dist, math.sin(ang), math.cos(ang)])

    data = Data(
        vit_emb=vit_embs,
        role_idx=role_idx,
        pos_features=positions,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.float) if src else torch.zeros((0, 5)),
        wall_angle=wall_angle_norm,
        num_nodes=n,
    )
    return data
