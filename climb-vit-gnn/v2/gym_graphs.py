"""Build v2 route graphs for gym routes from the cached DINOv2 embeddings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from gym_data import load_gym_routes  # noqa: E402
from kilter_data import DATA  # noqa: E402
from model_v2 import build_route  # noqa: E402

UI_ROLE = {"start": "start", "hand": "middle", "finish": "finish", "foot": "foot", "volume": "middle", "middle": "middle"}


class GymCache:
    def __init__(self, gym_root: str | Path, labeled_only: bool = True):
        self.routes = load_gym_routes(gym_root, labeled_only=labeled_only)
        self.store = np.load(DATA / "embeddings_v2" / "gym_embeddings.npz")
        self.meta = {m["id"]: m for m in json.load(open(DATA / "embeddings_v2" / "gym_meta.json"))}
        self.routes = [r for r in self.routes if f"{r['id']}/raw" in self.store]
        self.n_aug = 0
        while f"{self.routes[0]['id']}/aug{self.n_aug}" in self.store:
            self.n_aug += 1

    def graph(self, route: dict, key: str = "raw", flip: bool = False, jitter_px: float = 0.0, rng=None):
        """key: 'raw' | 'mask' | 'gray' | 'gray_mask' | 'augN'."""
        emb = torch.from_numpy(self.store[f"{route['id']}/{key}"])
        if key.startswith("aug"):
            flip = bool(self.store[f"{route['id']}/{key}_flip"][0]) or flip
        m = self.meta[route["id"]]
        xy = np.array([[h["tap_x"], m["h"] - h["tap_y"]] for h in route["holds"]], dtype=np.float32)
        if jitter_px > 0 and rng is not None:
            xy = xy + rng.normal(0, jitter_px, xy.shape).astype(np.float32)
        if flip:
            xy[:, 0] = m["w"] - xy[:, 0]
        roles = [UI_ROLE.get(h.get("role", "hand"), "middle") for h in route["holds"]]
        return build_route(emb, torch.from_numpy(xy), roles, route["wall_angle"], y=route.get("actual_diff"))
