"""Evaluate the *deployed* model on the labeled gym routes under different
crop preprocessing modes. No training involved: this isolates how much of
the gym error is an input-domain problem.

Usage:
    python v2/eval_gym.py --gym ../../gymdata --ckpt ../../hf/models/deployment_model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from gym_data import diff_to_v, load_gym_routes, load_image, summarize, v_index  # noqa: E402
from holds import crop_hold  # noqa: E402


def load_deployed(ckpt_path: str, hf_src: Path, device: str):
    sys.path.insert(0, str(hf_src))
    from src.inference.embed import HoldEmbedder  # noqa: E402
    from src.inference.graph import build_graph  # noqa: E402
    from src.inference.model import KilterViTGNN  # noqa: E402

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = KilterViTGNN(ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    embedder = HoldEmbedder("dinov2_vitb14", device=device)
    return model, embedder, build_graph


def metrics(pred_diff: np.ndarray, routes: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    true_v = np.array([r["actual_v"] for r in routes])
    true_d = np.array([r["actual_diff"] for r in routes], dtype=float)
    pred_v = np.array([v_index(diff_to_v(d)) for d in pred_diff])
    err_v = pred_v - true_v
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(routes), len(routes))
        boots.append(np.abs(err_v[idx]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(len(routes)),
        "mae_v": float(np.abs(err_v).mean()),
        "mae_v_ci95": [float(lo), float(hi)],
        "bias_v": float(err_v.mean()),
        "within1_v": float((np.abs(err_v) <= 1).mean()),
        "mae_diff": float(np.abs(pred_diff - true_d).mean()),
        "spearman": float(_spearman(pred_diff, true_d)),
    }


def _spearman(a, b):
    from scipy.stats import spearmanr

    return spearmanr(a, b).correlation


@torch.no_grad()
def predict_routes(model, embedder, build_graph, routes, images, mode, device):
    preds = []
    for r, img in zip(routes, images):
        crops = [crop_hold(img, h["tap_x"], h["tap_y"], mode=mode)[0] for h in r["holds"]]
        embs = embedder.embed_batch(crops)
        hold_data = [
            {"embedding": embs[i], "tap_x": h["tap_x"], "tap_y": h["tap_y"], "role": h["role"]}
            for i, h in enumerate(r["holds"])
        ]
        data = build_graph(hold_data, r["wall_angle"], img.shape)
        batch = next(iter(DataLoader([data], batch_size=1))).to(device)
        preds.append(float(model(batch).item()))
    return np.array(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gym", default=str(ROOT.parent.parent / "gymdata"))
    ap.add_argument("--ckpt", default=str(ROOT.parent.parent / "hf" / "models" / "deployment_model.pt"))
    ap.add_argument("--hf-src", default=str(ROOT.parent.parent / "hf"))
    ap.add_argument("--modes", default="raw,mask,gray,gray_mask")
    ap.add_argument("--out", default=str(ROOT / "v2" / "results" / "eval_gym_deployed.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    routes = load_gym_routes(args.gym)
    print(summarize(routes))
    images = [load_image(r) for r in routes]

    results = {}
    # Reference: what the deployed app actually said at submission time.
    recorded = np.array([r["old_predicted_diff"] for r in routes], dtype=float)
    results["recorded_app_prediction"] = metrics(recorded, routes)
    print("recorded_app_prediction", json.dumps(results["recorded_app_prediction"]))

    model, embedder, build_graph = load_deployed(args.ckpt, Path(args.hf_src), device)
    for mode in args.modes.split(","):
        pred = predict_routes(model, embedder, build_graph, routes, images, mode, device)
        results[f"deployed_{mode}"] = metrics(pred, routes)
        results[f"deployed_{mode}"]["pred_diff"] = [round(float(p), 2) for p in pred]
        print(f"deployed_{mode}", json.dumps({k: v for k, v in results[f'deployed_{mode}'].items() if k != 'pred_diff'}))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
