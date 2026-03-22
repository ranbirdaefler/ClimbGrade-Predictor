"""PyG Dataset using precomputed DINOv2 embeddings as node features."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

BOARD_MAX_X = 164.0
BOARD_MAX_Y = 176.0
MAX_ANGLE = 70.0

_ROLE_TO_IDX = {"start": 0, "middle": 1, "finish": 2, "foot_only": 3}

_FOOT_ONLY_IDX = 3
_MIDDLE_IDX = 1


class KilterViTDataset(TorchDataset):
    """Dataset of PyG graphs with DINOv2 embeddings as node features.

    Works with ``torch_geometric.loader.DataLoader``.

    Args:
        foot_drop_prob: probability of collapsing all foot_only roles
            to middle for a given sample (training-time augmentation).
            Set to 0.0 for validation/test.
    """

    def __init__(
        self,
        routes: list[dict],
        embeddings: np.ndarray,
        placement_to_crop_id: dict[int, int],
        crop_id_to_emb_idx: dict[int, int],
        foot_drop_prob: float = 0.0,
    ):
        super().__init__()
        self._graphs: list[Data] = []
        self.foot_drop_prob = foot_drop_prob

        for route in routes:
            graph = self._build_graph(
                route, embeddings, placement_to_crop_id, crop_id_to_emb_idx
            )
            if graph is not None:
                self._graphs.append(graph)

    @staticmethod
    def _build_graph(
        route: dict,
        embeddings: np.ndarray,
        placement_to_crop_id: dict[int, int],
        crop_id_to_emb_idx: dict[int, int],
    ) -> Data | None:
        holds = route["holds"]
        n = len(holds)
        angle_norm = route["angle"] / MAX_ANGLE

        vit_embs = []
        role_indices = []
        positions = []

        for h in holds:
            pid = h["placement_id"]
            crop_id = placement_to_crop_id.get(pid)
            if crop_id is None:
                return None
            emb_idx = crop_id_to_emb_idx.get(crop_id)
            if emb_idx is None:
                return None

            vit_embs.append(embeddings[emb_idx])
            role_indices.append(_ROLE_TO_IDX.get(h["role"], 1))
            positions.append([h["x"] / BOARD_MAX_X, h["y"] / BOARD_MAX_Y])

        # Fully connected edges (no self-loops)
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)

        edge_attr = []
        for s, d in zip(src, dst):
            dx = positions[d][0] - positions[s][0]
            dy = positions[d][1] - positions[s][1]
            dist = math.sqrt(dx * dx + dy * dy)
            ang = math.atan2(dy, dx)
            edge_attr.append([dx, dy, dist, math.sin(ang), math.cos(ang)])

        data = Data(
            vit_emb=torch.tensor(np.stack(vit_embs), dtype=torch.float),
            role_idx=torch.tensor(role_indices, dtype=torch.long),
            pos_features=torch.tensor(positions, dtype=torch.float),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float) if src else torch.zeros((0, 5)),
            wall_angle=torch.tensor([angle_norm], dtype=torch.float),
            y=torch.tensor([route["grade"]], dtype=torch.float),
            num_nodes=n,
        )
        data.uuid = route["uuid"]
        data.route_name = route.get("name", "")
        return data

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        data = self._graphs[idx]
        if self.foot_drop_prob > 0 and random.random() < self.foot_drop_prob:
            role = data.role_idx.clone()
            role[role == _FOOT_ONLY_IDX] = _MIDDLE_IDX
            data = data.clone()
            data.role_idx = role
        return data
