"""Pre-train ClimbGNN v2 on Kilter Board routes.

    python v2/train_v2.py --seed 0 --out v2/runs/v2_s0

Reports: Kilter val/test MAE (difficulty units and V-grades) and zero-shot
metrics on the labeled gym routes (never used for training here).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from gym_data import diff_to_v, v_index  # noqa: E402
from gym_graphs import GymCache  # noqa: E402
from kilter_data import DATA, TYPE_TO_IDX, hold_attribute_table, load_crop_index, load_routes  # noqa: E402
from model_v2 import ClimbGNNv2, build_route, route_geometry, soft_targets  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from model_v2 import ROLE_TO_IDX  # noqa: E402

torch.set_num_threads(4)

DEFAULT_CFG = {
    "d_hidden": 96, "proj_dim": 64, "proj_dropout": 0.15, "emb_dropout": 0.05,
    "d_role_emb": 8, "n_heads": 4, "n_layers": 3, "dropout": 0.15, "attn_dropout": 0.05,
    "lr": 1.5e-3, "weight_decay": 5e-4, "batch_size": 256, "epochs": 80, "patience": 15,
    "warmup_epochs": 3, "sigma": 1.0, "aux_weight": 0.3, "l1_weight": 0.1,
    "p_clean_view": 0.15, "p_flip": 0.5, "p_foot_role": 0.5, "p_hold_drop": 0.1,
}


class KilterV2Dataset(Dataset):
    def __init__(self, routes, views, hole_pos, pl_to_hole, attrs, cfg, train: bool):
        self.views = torch.from_numpy(views)  # [n_holes, K, 768]
        self.k = views.shape[1]
        self.cfg = cfg
        self.train = train
        self.items = []
        for r in routes:
            idx, roles, types, sizes, depths, xy = [], [], [], [], [], []
            ok = True
            for h in r["holds"]:
                hid = pl_to_hole.get(h["placement_id"])
                if hid is None or hid not in hole_pos:
                    ok = False
                    break
                idx.append(hole_pos[hid])
                roles.append(h["role"])
                a = attrs.get(h["placement_id"], {})
                types.append(TYPE_TO_IDX.get(a.get("type", "jug"), 0))
                sizes.append(int(a.get("size", 3)) - 1)
                depths.append(int(a.get("depth", 2)))
                xy.append([float(h["x"]), float(h["y"])])
            if not ok or len(idx) < 3:
                continue
            w = 0.5 + 0.5 * min(1.0, math.log10(max(r["ascents"], 1)) / 2.0)
            xy_t = torch.tensor(xy)
            xy_f = xy_t.clone(); xy_f[:, 0] = -xy_f[:, 0]
            self.items.append({
                "idx": torch.tensor(idx), "roles": roles, "types": types, "sizes": sizes, "depths": depths,
                "xy": xy_t, "y": float(r["grade"]), "w": w, "uuid": r["uuid"],
                "geom": route_geometry(xy_t), "geom_flip": route_geometry(xy_f),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        idx, roles, xy = it["idx"], list(it["roles"]), it["xy"].clone()
        types, sizes, depths = list(it["types"]), list(it["sizes"]), list(it["depths"])
        cfg = self.cfg
        geom = it["geom"]
        if self.train:
            n = len(roles)
            flip = random.random() < cfg["p_flip"]
            # drop a hold or two (users forget holds; setters add optional feet)
            if n > 5 and random.random() < cfg["p_hold_drop"]:
                keep = sorted(random.sample(range(n), n - random.randint(1, 2)))
                idx, xy = idx[keep], xy[keep]
                roles = [roles[k] for k in keep]; types = [types[k] for k in keep]
                sizes = [sizes[k] for k in keep]; depths = [depths[k] for k in keep]
                if flip:
                    xy[:, 0] = -xy[:, 0]
                geom = route_geometry(xy)
            elif flip:
                geom = it["geom_flip"]
            # Kilter has no foot role, but its small screw-on 'foot' holds are feet.
            roles = ["foot" if (t == TYPE_TO_IDX["foot"] and r == "middle" and random.random() < cfg["p_foot_role"]) else r
                     for r, t in zip(roles, types)]
            view = torch.where(torch.rand(len(idx)) < cfg["p_clean_view"], torch.zeros(len(idx), dtype=torch.long),
                               torch.randint(1, self.k, (len(idx),)))
        else:
            view = torch.zeros(len(idx), dtype=torch.long)
        emb = self.views[idx, view]
        g, ei, ea = geom
        data = Data(vit_emb=emb.float(), role_idx=torch.tensor([ROLE_TO_IDX.get(r, 1) for r in roles], dtype=torch.long),
                    geom=g, edge_index=ei, edge_attr=ea, wall_angle=torch.tensor([self._angle(i) / 70.0]),
                    y=torch.tensor([it["y"]]), w=torch.tensor([it["w"]]), num_nodes=len(roles))
        data.hold_type = torch.tensor(types); data.hold_size = torch.tensor(sizes); data.hold_depth = torch.tensor(depths)
        return data

    def _angle(self, i):
        return self._angles[i]


def make_dataset(routes, views, hole_pos, pl_to_hole, attrs, cfg, train):
    ds = KilterV2Dataset(routes, views, hole_pos, pl_to_hole, attrs, cfg, train)
    ang = {r["uuid"]: float(r["angle"]) for r in routes}
    ds._angles = [ang[it["uuid"]] for it in ds.items]
    return ds


def loss_fn(model, batch, cfg):
    logits, (t_l, s_l, d_l) = model(batch, return_aux=True)
    target = soft_targets(batch.y.view(-1), cfg["sigma"])
    ce = -(target * F.log_softmax(logits, dim=1)).sum(1)
    w = batch.w.view(-1)
    loss = (ce * w).sum() / w.sum()
    ev = ClimbGNNv2.expected(logits)
    loss = loss + cfg["l1_weight"] * ((ev - batch.y.view(-1)).abs() * w).sum() / w.sum()
    aux = 0.0
    for lg, tg in ((t_l, batch.hold_type), (s_l, batch.hold_size), (d_l, batch.hold_depth)):
        m = tg >= 0
        if m.any():
            aux = aux + F.cross_entropy(lg[m], tg[m])
    return loss + cfg["aux_weight"] * aux, loss.item()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ev, ys, logits_all = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        logits_all.append(logits.cpu())
        ev.append(ClimbGNNv2.expected(logits).cpu())
        if hasattr(batch, "y"):
            ys.append(batch.y.view(-1).cpu())
    return torch.cat(ev).numpy(), (torch.cat(ys).numpy() if ys else None), torch.cat(logits_all)


def kilter_metrics(pred, y):
    err = pred - y
    pv = np.array([v_index(diff_to_v(p)) for p in pred])
    yv = np.array([v_index(diff_to_v(t)) for t in y])
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "within1": float((np.abs(err) <= 1).mean()), "mae_v": float(np.abs(pv - yv).mean()),
            "within1_v": float((np.abs(pv - yv) <= 1).mean())}


def gym_metrics(pred, routes):
    from scipy.stats import spearmanr

    yv = np.array([r["actual_v"] for r in routes])
    yd = np.array([r["actual_diff"] for r in routes])
    pv = np.array([v_index(diff_to_v(p)) for p in pred])
    e = pv - yv
    return {"n": len(routes), "mae_v": float(np.abs(e).mean()), "bias_v": float(e.mean()),
            "within1_v": float((np.abs(e) <= 1).mean()), "mae_diff": float(np.abs(pred - yd).mean()),
            "spearman": float(spearmanr(pred, yd).correlation)}


def gym_zero_shot(model, device, gym_root, keys=("raw", "gray", "mask", "gray_mask")):
    cache = GymCache(gym_root)
    out = {}
    for key in keys:
        graphs = [cache.graph(r, key) for r in cache.routes]
        pred, _, _ = predict(model, DataLoader(graphs, batch_size=64), device)
        out[key] = gym_metrics(pred, cache.routes)
        out[key]["pred"] = [round(float(p), 2) for p in pred]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cfg", default="{}", help="JSON overrides of DEFAULT_CFG")
    ap.add_argument("--gym", default=str(ROOT.parent.parent / "gymdata"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = {**DEFAULT_CFG, **json.loads(args.cfg)}
    out_dir = Path(args.out or ROOT / "v2" / "runs" / f"v2_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    views = np.load(DATA / "embeddings_v2" / "kilter_views.npy")
    hole_ids = json.load(open(DATA / "embeddings_v2" / "kilter_hole_ids.json"))
    hole_pos = {h: i for i, h in enumerate(hole_ids)}
    pl_to_hole, _ = load_crop_index()
    train_r, val_r, test_r = load_routes("train"), load_routes("val"), load_routes("test")
    attrs = hold_attribute_table(train_r + val_r + test_r)

    tr = make_dataset(train_r, views, hole_pos, pl_to_hole, attrs, cfg, True)
    va = make_dataset(val_r, views, hole_pos, pl_to_hole, attrs, cfg, False)
    te = make_dataset(test_r, views, hole_pos, pl_to_hole, attrs, cfg, False)
    print(f"routes train={len(tr)} val={len(va)} test={len(te)} views={views.shape}")
    tl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=512)
    tel = DataLoader(te, batch_size=512)

    model = ClimbGNNv2(cfg).to(device)
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(tl)
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))

    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            loss, main_l = loss_fn(model, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tot += main_l * batch.num_graphs; n += batch.num_graphs
        pv, yv, _ = predict(model, vl, device)
        m = kilter_metrics(pv, yv)
        hist.append({"epoch": epoch, "train_loss": tot / n, **{f"val_{k}": v for k, v in m.items()}})
        if not args.quiet or epoch % 5 == 0:
            print(f"ep {epoch:3d} loss {tot / n:.3f} val MAE {m['mae']:.3f} within1 {m['within1']:.3f} ({time.time() - t0:.0f}s)")
        if m["mae"] < best - 1e-4:
            best, best_state, bad = m["mae"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("early stop"); break
    model.load_state_dict(best_state)
    pt, yt, _ = predict(model, tel, device)
    test_m = kilter_metrics(pt, yt)
    gym = gym_zero_shot(model, device, args.gym)
    print("KILTER TEST", json.dumps(test_m))
    for k, v in gym.items():
        print("GYM zero-shot", k, json.dumps({kk: vv for kk, vv in v.items() if kk != 'pred'}))
    torch.save({"model_state_dict": best_state, "cfg": cfg, "arch": "ClimbGNNv2", "seed": args.seed,
                "val_mae": best, "test": test_m}, out_dir / "model.pt")
    json.dump({"cfg": cfg, "history": hist, "val_mae": best, "test": test_m, "gym_zero_shot": gym},
              open(out_dir / "results.json", "w"), indent=2)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
