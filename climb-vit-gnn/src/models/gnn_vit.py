"""GATv2 GNN with DINOv2 visual embeddings as node features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool


class KilterViTGNN(nn.Module):
    """GNN that projects frozen DINOv2 embeddings and predicts route grade."""

    def __init__(self, config: dict):
        super().__init__()
        vit_cfg = config["vit"]
        model_cfg = config["model"]

        vit_dim = vit_cfg["embedding_dim"]
        proj_dim = model_cfg["proj_dim"]
        d_hidden = model_cfg["d_hidden"]
        n_heads = model_cfg["n_heads"]
        n_layers = model_cfg["n_layers"]
        edge_dim = model_cfg["edge_dim"]
        dropout = model_cfg["dropout"]
        d_role_emb = model_cfg["d_role_emb"]
        n_roles = model_cfg["n_roles"]

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Dropout(model_cfg["proj_dropout"]),
            nn.Linear(256, proj_dim),
        )

        self.role_emb = nn.Embedding(n_roles, d_role_emb)

        # proj_dim + d_role_emb + x(1) + y(1) + wall_angle(1)
        input_dim = proj_dim + d_role_emb + 3
        self.input_proj = nn.Linear(input_dim, d_hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    d_hidden,
                    d_hidden // n_heads,
                    heads=n_heads,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )
            self.norms.append(nn.LayerNorm(d_hidden))

        self.dropout = nn.Dropout(dropout)

        self.global_attn_pool = GlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(d_hidden, 1))
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data) -> torch.Tensor:
        vit_feat = self.vit_proj(data.vit_emb)
        role_e = self.role_emb(data.role_idx)

        wa = data.wall_angle
        if wa.dim() == 1:
            wa = wa.unsqueeze(-1)
        wall_angle_per_node = wa[data.batch]

        x = torch.cat([vit_feat, role_e, data.pos_features, wall_angle_per_node], dim=-1)
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, edge_attr=data.edge_attr)
            x = x + residual
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        x_mean = global_mean_pool(x, data.batch)
        x_attn = self.global_attn_pool(x, data.batch)

        wa_graph = data.wall_angle
        if wa_graph.dim() == 1:
            wa_graph = wa_graph.unsqueeze(-1)
        out = torch.cat([x_mean, x_attn, wa_graph], dim=-1)
        out = self.mlp(out)
        return out.squeeze(-1)
