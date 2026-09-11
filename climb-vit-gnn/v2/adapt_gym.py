"""Adapt pre-trained v2 models to real gym routes, with honest validation.

Strategies compared under repeated K-fold cross-validation over the labeled
gym routes (folds are grouped by photo so a route never leaks across folds):

  zero_shot      : pre-trained ensemble as-is
  calibrated     : zero_shot + affine correction (a * pred + b) fit on the
                   training folds
  finetune       : ensemble members fine-tuned on the training folds
                   (embedding-jitter + flip augmentation, small LR, few epochs),
                   then averaged
  finetune+cal   : finetune followed by affine calibration on training folds

Also reports a constant-median baseline and the recorded v1 app predictions.
Finally trains the chosen strategy on ALL labeled routes and exports it.

    python v2/adapt_gym.py --runs v2/runs/v2_s0 v2/runs/v2_s1 ... --out v2/runs/adapt
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from gym_data import diff_to_v, v_index  # noqa: E402
from gym_graphs import GymCache  # noqa: E402
from model_v2 import ClimbGNNv2, soft_targets  # noqa: E402
from train_v2 import gym_metrics, predict  # noqa: E402


def load_models(run_dirs, device):
    models = []
    for d in run_dirs:
        ck = torch.load(Path(d) / "model.pt", map_location=device, weights_only=False)
        m = ClimbGNNv2(ck["cfg"]).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        models.append(m)
    return models


def ensemble_predict(models, graphs, device):
    preds = []
    for m in models:
        p, _, _ = predict(m, DataLoader(graphs, batch_size=64), device)
        preds.append(p)
    return np.mean(preds, axis=0)


def fit_affine(pred, y, shrink: float = 0.5):
    """Ridge-shrunk affine map pred -> y. shrink pulls slope toward 1."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    pm, ym = pred.mean(), y.mean()
    var = ((pred - pm) ** 2).sum()
    cov = ((pred - pm) * (y - ym)).sum()
    a = (cov + shrink * var) / (var + shrink * var) if var > 0 else 1.0
    a = float(np.clip(a, 0.3, 1.5))
    b = float(ym - a * pm)
    return a, b


def finetune(model, cache, train_routes, device, cfg, seed):
    """Fine-tune all weights briefly on the gym training routes."""
    random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = copy.deepcopy(model)
    proj_params = list(m.vit_proj.parameters())
    proj_ids = {id(p) for p in proj_params}
    other = [p for p in m.parameters() if id(p) not in proj_ids]
    if cfg["ft_proj_lr"] <= 0:
        for p in proj_params:
            p.requires_grad = False
        groups = [{"params": other, "lr": cfg["ft_lr"]}]
    else:
        groups = [{"params": other, "lr": cfg["ft_lr"]}, {"params": proj_params, "lr": cfg["ft_proj_lr"]}]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(groups, weight_decay=1e-3)
    key = cfg.get("key", "raw")
    keys = [key] + ([f"aug{i}" for i in range(cache.n_aug)] if key == "raw" else [])
    m.train()
    for _ in range(cfg["ft_epochs"]):
        graphs = []
        for r in train_routes:
            key = keys[rng.integers(0, len(keys))]
            graphs.append(cache.graph(r, key, flip=bool(rng.random() < 0.5), jitter_px=cfg["ft_jitter_px"], rng=rng))
        for batch in DataLoader(graphs, batch_size=16, shuffle=True):
            batch = batch.to(device)
            opt.zero_grad()
            logits = m(batch)
            tgt = soft_targets(batch.y.view(-1), cfg["ft_sigma"])
            loss = -(tgt * F.log_softmax(logits, 1)).sum(1).mean()
            loss = loss + 0.2 * (ClimbGNNv2.expected(logits) - batch.y.view(-1)).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    m.eval()
    return m


def grouped_folds(routes, k, seed):
    groups = sorted({r["image_hash"] for r in routes})
    rng = random.Random(seed)
    rng.shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    return [[r for r in routes if fold_of[r["image_hash"]] == f] for f in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gym", default=str(ROOT.parent.parent / "gymdata"))
    ap.add_argument("--out", default=str(ROOT / "v2" / "runs" / "adapt"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--ft-epochs", type=int, default=25)
    ap.add_argument("--ft-sigma", type=float, default=1.5)
    ap.add_argument("--ft-jitter-px", type=float, default=6.0)
    ap.add_argument("--key", default="raw", help="crop mode used for inference")
    ap.add_argument("--ft-proj-lr", type=float, default=0.0, help="LR for the visual projection (0 = frozen)")
    args = ap.parse_args()
    cfg = {"ft_lr": args.ft_lr, "ft_epochs": args.ft_epochs, "ft_sigma": args.ft_sigma, "ft_jitter_px": args.ft_jitter_px,
           "ft_proj_lr": args.ft_proj_lr, "key": args.key}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cache = GymCache(args.gym)
    routes = cache.routes
    models = load_models(args.runs, device)
    print(f"{len(routes)} labeled gym routes, {len(models)} ensemble members, key={args.key}")

    strategies = ["constant_median", "v1_recorded", "ridge_embed", "zero_shot", "calibrated", "finetune", "finetune+cal"]
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    mean_emb = {r["id"]: cache.store[f"{r['id']}/{args.key}"].mean(0) for r in routes}

    def ridge_fit_predict(train_routes, test_routes):
        Xtr = np.array([mean_emb[r["id"]] for r in train_routes]); Xte = np.array([mean_emb[r["id"]] for r in test_routes])
        pc = PCA(n_components=8).fit(Xtr)
        Xtr, Xte = pc.transform(Xtr), pc.transform(Xte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        mdl = Ridge(alpha=10.0).fit((Xtr - mu) / sd, np.array([r["actual_diff"] for r in train_routes]))
        return mdl.predict((Xte - mu) / sd)
    oof = {s: {r["id"]: [] for r in routes} for s in strategies}
    for rep in range(args.repeats):
        folds = grouped_folds(routes, args.folds, seed=100 + rep)
        for f, test_routes in enumerate(folds):
            train_routes = [r for g, fr in enumerate(folds) if g != f for r in fr]
            test_graphs = [cache.graph(r, args.key) for r in test_routes]
            train_graphs = [cache.graph(r, args.key) for r in train_routes]
            y_tr = np.array([r["actual_diff"] for r in train_routes])
            med = float(np.median(y_tr))
            zs_te = ensemble_predict(models, test_graphs, device)
            zs_tr = ensemble_predict(models, train_graphs, device)
            a, b = fit_affine(zs_tr, y_tr)
            ft_models = [finetune(m, cache, train_routes, device, cfg, seed=rep * 10 + i) for i, m in enumerate(models)]
            ft_te = ensemble_predict(ft_models, test_graphs, device)
            ft_tr = ensemble_predict(ft_models, train_graphs, device)
            a2, b2 = fit_affine(ft_tr, y_tr)
            rg_te = ridge_fit_predict(train_routes, test_routes)
            for r, zs, ft, rg in zip(test_routes, zs_te, ft_te, rg_te):
                oof["ridge_embed"][r["id"]].append(float(rg))
                oof["constant_median"][r["id"]].append(med)
                oof["v1_recorded"][r["id"]].append(float(r["old_predicted_diff"]))
                oof["zero_shot"][r["id"]].append(float(zs))
                oof["calibrated"][r["id"]].append(float(a * zs + b))
                oof["finetune"][r["id"]].append(float(ft))
                oof["finetune+cal"][r["id"]].append(float(a2 * ft + b2))
            print(f"rep {rep} fold {f}: n_test={len(test_routes)} affine a={a:.2f} b={b:.2f} | ft affine a={a2:.2f} b={b2:.2f}")

    results = {}
    for s in strategies:
        pred = np.array([np.mean(oof[s][r["id"]]) for r in routes])
        results[s] = gym_metrics(pred, routes)
        # bootstrap CI on MAE_v
        yv = np.array([r["actual_v"] for r in routes])
        pv = np.array([v_index(diff_to_v(p)) for p in pred])
        rng = np.random.default_rng(0)
        boots = [np.abs((pv - yv)[rng.integers(0, len(yv), len(yv))]).mean() for _ in range(2000)]
        results[s]["mae_v_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        results[s]["oof_pred"] = [round(float(p), 2) for p in pred]
        print(f"{s:16s} MAE_v {results[s]['mae_v']:.2f} [{results[s]['mae_v_ci95'][0]:.2f},{results[s]['mae_v_ci95'][1]:.2f}]"
              f" bias {results[s]['bias_v']:+.2f} within1 {results[s]['within1_v']:.2f} spearman {results[s]['spearman']:.2f}")
    # paired comparison: fraction of bootstrap resamples where a strategy beats v1
    yv = np.array([r["actual_v"] for r in routes])
    e1 = np.abs(np.array([v_index(diff_to_v(p)) for p in results["v1_recorded"]["oof_pred"]]) - yv)
    for s in ("zero_shot", "calibrated", "finetune", "finetune+cal"):
        es = np.abs(np.array([v_index(diff_to_v(p)) for p in results[s]["oof_pred"]]) - yv)
        rng = np.random.default_rng(1)
        wins = np.mean([(es[idx].mean() < e1[idx].mean()) for idx in (rng.integers(0, len(yv), len(yv)) for _ in range(2000))])
        results[s]["p_better_than_v1"] = float(wins)
        print(f"P({s} beats v1 on MAE) = {wins:.3f}")
    json.dump({"cfg": cfg, "key": args.key, "results": results, "ids": [r["id"] for r in routes],
               "actual_v": [r["actual_v"] for r in routes]}, open(out / "cv_results.json", "w"), indent=2)

    # ── Final: fit the best strategy on all labeled routes ────────────────
    order = ["finetune+cal", "finetune", "calibrated", "zero_shot"]
    best = min(order, key=lambda s: results[s]["mae_v"])
    print("best strategy:", best)
    all_graphs = [cache.graph(r, args.key) for r in routes]
    y_all = np.array([r["actual_diff"] for r in routes])
    final_models = models
    if best.startswith("finetune"):
        final_models = [finetune(m, cache, routes, device, cfg, seed=999 + i) for i, m in enumerate(models)]
    a, b = (1.0, 0.0)
    if best.endswith("cal") or best == "calibrated":
        a, b = fit_affine(ensemble_predict(final_models, all_graphs, device), y_all)
    torch.save({
        "arch": "ClimbGNNv2",
        "cfg": final_models[0].cfg,
        "members": [m.state_dict() for m in final_models],
        "affine": [a, b],
        "strategy": best,
        "crop_mode": args.key,
        "cv": {s: {k: v for k, v in results[s].items() if k != "oof_pred"} for s in strategies},
    }, out / "deployment_v2.pt")
    print("saved", out / "deployment_v2.pt", "affine", a, b)


if __name__ == "__main__":
    main()
