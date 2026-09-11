"""ClimbGNN v2: route-graph grade model designed to transfer from board renders
to gym photos.

Changes vs v1 (KilterViTGNN):
  * Geometry is scale-invariant: positions are centred on the route and scaled
    by the route's extent; edge distances are normalised by the median
    nearest-neighbour distance. Works for any photo framing.
  * Roles: start / middle / finish / foot, all trained.
  * Auxiliary heads predict hold type / size / depth from the projected
    visual embedding, so the projection keeps physically meaningful
    information instead of memorising Kilter render identity.
  * Output is a distribution over 21 difficulty bins (10..30) trained with
    soft-label cross-entropy; the expected value is the point estimate and
    the distribution gives a calibrated grade range.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

ROLES = ["start", "middle", "finish", "foot"]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}
N_BINS = 21
BIN_CENTRES = torch.arange(10, 31, dtype=torch.float)  # 10..30
N_TYPES, N_SIZES, N_DEPTHS = 5, 5, 4
GEOM_DIM = 6          # x, y (route-normalised), x, y (rank-normalised), local density, n_holds
EDGE_DIM = 6          # dx, dy, dist (NN-normalised), sin, cos, dist (span-normalised)


def route_geometry(xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """xy: [n, 2] with y pointing UP. Returns (node_geom [n, GEOM_DIM],
    edge_index [2, n(n-1)], edge_attr [n(n-1), EDGE_DIM])."""
    n = xy.shape[0]
    centre = xy.mean(0, keepdim=True)
    span = (xy.max(0).values - xy.min(0).values).max().clamp(min=1e-6)
    pos = (xy - centre) / span                      # roughly in [-0.5, 0.5]
    # rank-normalised order (bottom = 0, top = 1): robust to outliers
    rank = torch.zeros_like(xy)
    for d in range(2):
        order = xy[:, d].argsort()
        rank[order, d] = torch.linspace(0, 1, n) if n > 1 else torch.tensor([0.5])
    diff = xy[:, None, :] - xy[None, :, :]          # [n, n, 2] (j - i)
    dist = diff.norm(dim=-1)
    eye = torch.eye(n, dtype=torch.bool)
    nn_dist = dist.masked_fill(eye, float("inf")).min(1).values if n > 1 else torch.ones(n)
    nn_med = nn_dist.median().clamp(min=1e-6)
    density = (dist < 2 * nn_med).float().sum(1) - 1  # neighbours within 2 NN-dists
    density = density / max(n - 1, 1)
    geom = torch.cat([pos, rank, density[:, None], torch.full((n, 1), math.log(n) / 4)], dim=1)

    ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    off = ii != jj
    ei = torch.stack([ii[off], jj[off]])
    d = diff[ei[0], ei[1]] / nn_med                 # NN-normalised displacement
    dn = d.norm(dim=-1, keepdim=True)
    ang = torch.atan2(d[:, 1], d[:, 0])
    span_d = dist[ei[0], ei[1], None] / span
    ea = torch.cat([d.clamp(-6, 6) / 3, dn.clamp(0, 8) / 4, ang.sin()[:, None], ang.cos()[:, None], span_d], dim=1)
    return geom, ei, ea


def build_route(
    embeddings: torch.Tensor,
    xy_up: torch.Tensor,
    roles: list[str],
    wall_angle: float,
    y: float | None = None,
    hold_type: list[int] | None = None,
    hold_size: list[int] | None = None,
    hold_depth: list[int] | None = None,
    weight: float = 1.0,
) -> Data:
    geom, ei, ea = route_geometry(xy_up)
    n = len(roles)
    data = Data(
        vit_emb=embeddings.float(),
        role_idx=torch.tensor([ROLE_TO_IDX.get(r, 1) for r in roles], dtype=torch.long),
        geom=geom,
        edge_index=ei,
        edge_attr=ea,
        wall_angle=torch.tensor([wall_angle / 70.0], dtype=torch.float),
        num_nodes=n,
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
        data.w = torch.tensor([weight], dtype=torch.float)
    data.hold_type = torch.tensor(hold_type if hold_type is not None else [-1] * n, dtype=torch.long)
    data.hold_size = torch.tensor(hold_size if hold_size is not None else [-1] * n, dtype=torch.long)
    data.hold_depth = torch.tensor(hold_depth if hold_depth is not None else [-1] * n, dtype=torch.long)
    return data


def soft_targets(y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    centres = BIN_CENTRES.to(y.device)
    logits = -0.5 * ((y[:, None] - centres[None, :]) / sigma) ** 2
    return torch.softmax(logits, dim=1)


class ClimbGNNv2(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_hidden"]
        proj = cfg["proj_dim"]
        heads = cfg["n_heads"]
        self.cfg = cfg
        self.vit_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(cfg["proj_dropout"]),
            nn.Linear(256, proj),
        )
        self.emb_dropout = nn.Dropout(cfg.get("emb_dropout", 0.0))
        self.type_head = nn.Linear(proj, N_TYPES)
        self.size_head = nn.Linear(proj, N_SIZES)
        self.depth_head = nn.Linear(proj, N_DEPTHS)
        self.role_emb = nn.Embedding(len(ROLES), cfg["d_role_emb"])
        self.geom_proj = nn.Linear(GEOM_DIM + 1, 32)
        self.input_proj = nn.Linear(proj + cfg["d_role_emb"] + 32, d)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg["n_layers"]):
            self.convs.append(GATv2Conv(d, d // heads, heads=heads, edge_dim=EDGE_DIM, add_self_loops=False, dropout=cfg.get("attn_dropout", 0.0)))
            self.norms.append(nn.LayerNorm(d))
        self.dropout = nn.Dropout(cfg["dropout"])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1)))
        self.head = nn.Sequential(
            nn.Linear(d * 3 + 1, 128), nn.GELU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, N_BINS),
        )

    def node_features(self, data):
        v = self.vit_proj(self.emb_dropout(data.vit_emb))
        return v

    def forward(self, data, return_aux: bool = False):
        v = self.node_features(data)
        wa = data.wall_angle.view(-1, 1)
        wa_node = wa[data.batch]
        g = self.geom_proj(torch.cat([data.geom, wa_node], dim=1))
        x = self.input_proj(torch.cat([v, self.role_emb(data.role_idx), g], dim=1))
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, data.edge_index, edge_attr=data.edge_attr)
            x = norm(x + self.dropout(F.gelu(h)))
        pooled = torch.cat([
            global_mean_pool(x, data.batch),
            global_max_pool(x, data.batch),
            self.attn_pool(x, data.batch),
            wa,
        ], dim=1)
        logits = self.head(pooled)
        if return_aux:
            return logits, (self.type_head(v), self.size_head(v), self.depth_head(v))
        return logits

    @staticmethod
    def expected(logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=1)
        return (p * BIN_CENTRES.to(logits.device)[None, :]).sum(1)

    @staticmethod
    def quantiles(logits: torch.Tensor, qs=(0.16, 0.84)) -> torch.Tensor:
        p = torch.softmax(logits, dim=1)
        cdf = p.cumsum(1)
        centres = BIN_CENTRES.to(logits.device)
        out = []
        for q in qs:
            idx = (cdf < q).sum(1).clamp(max=N_BINS - 1)
            out.append(centres[idx])
        return torch.stack(out, dim=1)
