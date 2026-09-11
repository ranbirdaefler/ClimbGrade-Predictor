"""ClimbGNN v2 inference: ensemble of route-graph models with scale-invariant
geometry, a distributional grade head, and a gym-calibrated affine map.

Mirrors climb-vit-gnn/v2/model_v2.py (kept dependency-free of the training
code so the Space only ships what it runs).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

ROLES = ["start", "middle", "finish", "foot"]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}
UI_ROLE = {"start": "start", "hand": "middle", "finish": "finish", "foot": "foot", "volume": "middle", "middle": "middle"}
N_BINS = 21
BIN_CENTRES = torch.arange(10, 31, dtype=torch.float)
N_TYPES, N_SIZES, N_DEPTHS = 5, 5, 4
GEOM_DIM, EDGE_DIM = 6, 6


def route_geometry(xy: torch.Tensor):
    n = xy.shape[0]
    centre = xy.mean(0, keepdim=True)
    span = (xy.max(0).values - xy.min(0).values).max().clamp(min=1e-6)
    pos = (xy - centre) / span
    rank = torch.zeros_like(xy)
    for d in range(2):
        order = xy[:, d].argsort()
        rank[order, d] = torch.linspace(0, 1, n) if n > 1 else torch.tensor([0.5])
    diff = xy[:, None, :] - xy[None, :, :]
    dist = diff.norm(dim=-1)
    eye = torch.eye(n, dtype=torch.bool)
    nn_dist = dist.masked_fill(eye, float("inf")).min(1).values if n > 1 else torch.ones(n)
    nn_med = nn_dist.median().clamp(min=1e-6)
    density = ((dist < 2 * nn_med).float().sum(1) - 1) / max(n - 1, 1)
    geom = torch.cat([pos, rank, density[:, None], torch.full((n, 1), math.log(n) / 4)], dim=1)
    ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    off = ii != jj
    ei = torch.stack([ii[off], jj[off]])
    d = diff[ei[0], ei[1]] / nn_med
    dn = d.norm(dim=-1, keepdim=True)
    ang = torch.atan2(d[:, 1], d[:, 0])
    span_d = dist[ei[0], ei[1], None] / span
    ea = torch.cat([d.clamp(-6, 6) / 3, dn.clamp(0, 8) / 4, ang.sin()[:, None], ang.cos()[:, None], span_d], dim=1)
    return geom, ei, ea


def build_route(embeddings: torch.Tensor, xy_up: torch.Tensor, roles: list[str], wall_angle: float) -> Data:
    geom, ei, ea = route_geometry(xy_up)
    return Data(
        vit_emb=embeddings.float(),
        role_idx=torch.tensor([ROLE_TO_IDX.get(r, 1) for r in roles], dtype=torch.long),
        geom=geom, edge_index=ei, edge_attr=ea,
        wall_angle=torch.tensor([wall_angle / 70.0], dtype=torch.float),
        num_nodes=len(roles),
    )


class ClimbGNNv2(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, proj, heads = cfg["d_hidden"], cfg["proj_dim"], cfg["n_heads"]
        self.cfg = cfg
        self.vit_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Dropout(cfg["proj_dropout"]), nn.Linear(256, proj))
        self.emb_dropout = nn.Dropout(cfg.get("emb_dropout", 0.0))
        self.type_head = nn.Linear(proj, N_TYPES)
        self.size_head = nn.Linear(proj, N_SIZES)
        self.depth_head = nn.Linear(proj, N_DEPTHS)
        self.role_emb = nn.Embedding(len(ROLES), cfg["d_role_emb"])
        self.geom_proj = nn.Linear(GEOM_DIM + 1, 32)
        self.input_proj = nn.Linear(proj + cfg["d_role_emb"] + 32, d)
        self.convs = nn.ModuleList([GATv2Conv(d, d // heads, heads=heads, edge_dim=EDGE_DIM, add_self_loops=False, dropout=cfg.get("attn_dropout", 0.0)) for _ in range(cfg["n_layers"])])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg["n_layers"])])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1)))
        self.head = nn.Sequential(nn.Linear(d * 3 + 1, 128), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, N_BINS))

    def forward(self, data):
        v = self.vit_proj(self.emb_dropout(data.vit_emb))
        wa = data.wall_angle.view(-1, 1)
        g = self.geom_proj(torch.cat([data.geom, wa[data.batch]], dim=1))
        x = self.input_proj(torch.cat([v, self.role_emb(data.role_idx), g], dim=1))
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + self.dropout(F.gelu(conv(x, data.edge_index, edge_attr=data.edge_attr))))
        pooled = torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch), self.attn_pool(x, data.batch), wa], dim=1)
        return self.head(pooled)


class EnsembleV2:
    """Loads a deployment_v2.pt bundle and predicts a difficulty distribution."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.device = torch.device(device)
        self.cfg = ck["cfg"]
        self.affine = tuple(ck.get("affine", (1.0, 0.0)))
        self.crop_mode = ck.get("crop_mode", "raw")
        self.strategy = ck.get("strategy", "?")
        self.members = []
        for sd in ck["members"]:
            m = ClimbGNNv2(self.cfg).to(self.device)
            m.load_state_dict(sd)
            m.eval()
            self.members.append(m)

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor, taps_xy: list[tuple[int, int]], roles: list[str], wall_angle: float, image_hw: tuple[int, int]) -> dict:
        h, _w = image_hw
        xy = torch.tensor([[float(x), float(h - y)] for x, y in taps_xy])
        data = build_route(embeddings.cpu(), xy, [UI_ROLE.get(r, "middle") for r in roles], wall_angle)
        batch = Batch.from_data_list([data]).to(self.device)
        probs = torch.stack([torch.softmax(m(batch), dim=1)[0] for m in self.members]).mean(0).cpu()
        centres = BIN_CENTRES
        a, b = self.affine
        expected = float((probs * centres).sum()) * a + b
        # Distribution quantiles, mapped through the same affine correction.
        cdf = probs.cumsum(0)
        q_lo = float(centres[int((cdf < 0.16).sum().clamp(max=N_BINS - 1))]) * a + b
        q_hi = float(centres[int((cdf < 0.84).sum().clamp(max=N_BINS - 1))]) * a + b
        # Never report a band narrower than +-1 difficulty unit around the point estimate.
        q_lo, q_hi = min(q_lo, expected - 1.0), max(q_hi, expected + 1.0)
        return {"difficulty": expected, "low": q_lo, "high": q_hi, "probs": probs.numpy()}
