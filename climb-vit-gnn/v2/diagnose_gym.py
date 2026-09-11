"""How much grade signal is in the gym data at all? Leave-one-photo-out ridge
regression on cheap features, to set expectations for any model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from gym_data import diff_to_v, v_index  # noqa: E402
from gym_graphs import GymCache  # noqa: E402

cache = GymCache(ROOT.parent.parent / "gymdata")
routes = cache.routes
y = np.array([r["actual_diff"] for r in routes])
yv = np.array([r["actual_v"] for r in routes])
groups = np.array([r["image_hash"] for r in routes])


def geom_feats(r):
    m = cache.meta[r["id"]]
    xy = np.array([[h["tap_x"], m["h"] - h["tap_y"]] for h in r["holds"]], float)
    diag = np.hypot(m["w"], m["h"])
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    span = np.ptp(xy, axis=0)
    roles = [h.get("role", "hand") for h in r["holds"]]
    return [len(r["holds"]), r["wall_angle"] / 70, roles.count("foot") / len(roles), roles.count("volume") / len(roles),
            nn.mean() / diag, nn.max() / diag, span[1] / diag, span[0] / diag, (nn / span.max()).mean(), np.log(len(r["holds"]))]


G = np.array([geom_feats(r) for r in routes])
E = np.array([cache.store[f"{r['id']}/raw"].mean(0) for r in routes])
E_gm = np.array([cache.store[f"{r['id']}/gray_mask"].mean(0) for r in routes])
v1 = np.array([r["old_predicted_diff"] for r in routes])[:, None]
v2 = None
p = ROOT / "v2" / "runs" / "v2_s0" / "results.json"
if p.exists():
    res = json.load(open(p))["gym_zero_shot"]
    v2 = np.array([res["raw"]["pred"], res["gray_mask"]["pred"]]).T


def loo(X, alpha=1.0, pca=None):
    preds = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        Xtr, Xte = X[tr], X[te]
        if pca:
            pc = PCA(n_components=min(pca, tr.sum() - 1)).fit(Xtr)
            Xtr, Xte = pc.transform(Xtr), pc.transform(Xte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        m = Ridge(alpha=alpha).fit((Xtr - mu) / sd, y[tr])
        preds[te] = m.predict((Xte - mu) / sd)
    pv = np.array([v_index(diff_to_v(q)) for q in preds])
    return float(np.abs(pv - yv).mean()), float(spearmanr(preds, y).correlation), float((np.abs(pv - yv) <= 1).mean())


print(f"{len(routes)} routes; constant-median MAE_v = {np.abs(np.median(yv) - yv).mean():.2f}")
rows = [
    ("angle only", G[:, [1]], 1.0, None),
    ("n_holds only", G[:, [0]], 1.0, None),
    ("geometry (10 feats)", G, 3.0, None),
    ("mean DINOv2 raw (PCA8)", E, 10.0, 8),
    ("mean DINOv2 gray_mask (PCA8)", E_gm, 10.0, 8),
    ("geometry + DINOv2 raw PCA8", np.hstack([G, E]), 10.0, None),
    ("v1 prediction only", v1, 1.0, None),
]
if v2 is not None:
    rows += [("v2 zero-shot preds only", v2, 1.0, None), ("geometry + v2 preds", np.hstack([G, v2]), 3.0, None)]
for name, X, a, pca in rows:
    if name == "geometry + DINOv2 raw PCA8":
        # pca on the embedding block only
        pc = PCA(n_components=8).fit(E)
        X = np.hstack([G, pc.transform(E)])
    mae, rho, w1 = loo(X, a, pca)
    print(f"{name:34s} LOO MAE_v {mae:.2f}  spearman {rho:+.2f}  within1 {w1:.2f}")
print("raw correlations with grade:")
for i, n in enumerate(["n_holds", "angle", "feet_frac", "vol_frac", "nn_mean", "nn_max", "span_y", "span_x", "nn/span", "log_n"]):
    print(f"  {n:10s} rho={spearmanr(G[:, i], y).correlation:+.2f}")
