"""Precompute DINOv2 embeddings of Kilter hold crops under photo-like
augmentations, and of gym hold crops under evaluation/adaptation variants.

Kilter renders are grey shapes on a neutral background; real gym holds are
coloured, lit, on textured walls, photographed at odd angles. We simulate
that on the renders so the GNN learns from embeddings that look like what it
will see at inference. Output:

    data/embeddings_v2/kilter_views.npy     [n_holes, K, 768]  view 0 = clean
    data/embeddings_v2/kilter_hole_ids.json
    data/embeddings_v2/gym_embeddings.npz   per route, per mode, per aug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from gym_data import load_gym_routes, load_image  # noqa: E402
from holds import crop_hold  # noqa: E402
from kilter_data import DATA, crop_background_mask, load_crop, load_crop_index  # noqa: E402

OUT = DATA / "embeddings_v2"


def dinov2(device: str):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, tf


@torch.no_grad()
def embed_pils(model, tf, pils: list[Image.Image], device: str, bs: int = 96) -> np.ndarray:
    out = []
    for i in range(0, len(pils), bs):
        batch = torch.stack([tf(p) for p in pils[i:i + bs]]).to(device)
        out.append(model(batch).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


# ── Photo-like augmentation of a grey Kilter render ──────────────────
def _synthetic_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.integers(0, 4)
    base = rng.integers(60, 230, size=3).astype(np.float32)
    if kind == 0:  # flat wall colour
        bg = np.tile(base, (h, w, 1))
    elif kind == 1:  # gradient (lighting)
        g = np.linspace(-1, 1, w)[None, :, None] * rng.uniform(-40, 40)
        v = np.linspace(-1, 1, h)[:, None, None] * rng.uniform(-40, 40)
        bg = np.tile(base, (h, w, 1)) + g + v
    elif kind == 2:  # plywood/paint texture: low-frequency noise
        noise = rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        bg = np.tile(base, (h, w, 1)) + noise[:, :, None] * rng.uniform(5, 25)
    else:  # speckle (t-nut holes, chalk)
        bg = np.tile(base, (h, w, 1)) + rng.normal(0, 12, (h, w, 1))
        for _ in range(rng.integers(0, 6)):
            cv2.circle(bg, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(2, 6)), (20, 20, 20), -1)
    return np.clip(bg, 0, 255).astype(np.uint8)


def augment_render(crop: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(crop).copy()
    h, w = arr.shape[:2]
    bg_mask = crop_background_mask(arr)
    fg = 1 - bg_mask

    # Colourise the grey hold: keep shading (value), set random hue/saturation.
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = rng.uniform(0, 180)
    sat = rng.uniform(0.0, 0.95) * 255 if rng.random() > 0.15 else rng.uniform(0, 0.15) * 255  # some holds are white/black/wood
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = sat
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.6, 1.35) + rng.uniform(-25, 25), 0, 255)
    coloured = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    bg = _synthetic_background(h, w, rng)
    fg3 = fg[:, :, None].astype(np.float32)
    # soften the matte a bit so the edge is not a hard cut
    fg3 = cv2.GaussianBlur(fg3, (0, 0), rng.uniform(0.5, 1.5))[:, :, None]
    out = coloured.astype(np.float32) * fg3 + bg.astype(np.float32) * (1 - fg3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Geometry: small rotation / scale / perspective, like a phone photo.
    ang = rng.uniform(-25, 25)
    scale = rng.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[:, 2] += rng.uniform(-0.08, 0.08, size=2) * w
    border = tuple(int(v) for v in bg[0, 0])
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    if rng.random() < 0.5:
        out = out[:, ::-1]  # mirror

    pil = Image.fromarray(out)
    if rng.random() < 0.6:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 2.0)))
    if rng.random() < 0.5:  # jpeg-ish noise
        a = np.asarray(pil).astype(np.float32) + rng.normal(0, rng.uniform(2, 10), (h, w, 3))
        pil = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return pil


def build_kilter_views(device: str, k_views: int, seed: int) -> None:
    model, tf = dinov2(device)
    _, hole_ids = load_crop_index()
    rng = np.random.default_rng(seed)
    views = np.zeros((len(hole_ids), k_views, 768), np.float32)
    for i, hid in enumerate(tqdm(hole_ids, desc="kilter views")):
        crop = load_crop(hid)
        pils = [crop] + [augment_render(crop, rng) for _ in range(k_views - 1)]
        views[i] = embed_pils(model, tf, pils, device)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "kilter_views.npy", views)
    json.dump(hole_ids, open(OUT / "kilter_hole_ids.json", "w"))
    # Save a contact sheet so the augmentation can be eyeballed.
    sheet = Image.new("RGB", (224 * 8, 224 * 4))
    rng2 = np.random.default_rng(1)
    for r in range(4):
        crop = load_crop(hole_ids[int(rng2.integers(0, len(hole_ids)))])
        sheet.paste(crop, (0, r * 224))
        for c in range(1, 8):
            sheet.paste(augment_render(crop, rng2), (c * 224, r * 224))
    sheet.save(OUT / "augmentation_sheet.jpg", quality=80)
    print("saved", views.shape)


def build_gym_embeddings(device: str, gym_root: str, n_aug: int, seed: int) -> None:
    """For every labeled gym route: embeddings per hold for
    modes raw/mask/gray/gray_mask, plus n_aug jittered+flipped raw variants."""
    model, tf = dinov2(device)
    routes = load_gym_routes(gym_root, labeled_only=False)
    rng = np.random.default_rng(seed)
    store: dict[str, np.ndarray] = {}
    meta = []
    for r in tqdm(routes, desc="gym routes"):
        img = load_image(r)
        h, w = img.shape[:2]
        for mode in ("raw", "mask", "gray", "gray_mask"):
            pils = [crop_hold(img, hd["tap_x"], hd["tap_y"], mode=mode)[0] for hd in r["holds"]]
            store[f"{r['id']}/{mode}"] = embed_pils(model, tf, pils, device)
        # Augmented variants: crop-window jitter + random mirror + colour jitter.
        for a in range(n_aug):
            pils = []
            flip = rng.random() < 0.5
            for hd in r["holds"]:
                jit = (int(rng.integers(-12, 13)), int(rng.integers(-12, 13)))
                p, _ = crop_hold(img, hd["tap_x"], hd["tap_y"], mode="raw", jitter=jit)
                arr = np.asarray(p).astype(np.float32)
                arr = arr * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
                if flip:
                    arr = arr[:, ::-1]
                if rng.random() < 0.4:
                    arr = cv2.GaussianBlur(np.clip(arr, 0, 255).astype(np.uint8), (0, 0), rng.uniform(0.3, 1.5)).astype(np.float32)
                pils.append(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
            store[f"{r['id']}/aug{a}"] = embed_pils(model, tf, pils, device)
            store[f"{r['id']}/aug{a}_flip"] = np.array([flip], dtype=np.bool_)
        meta.append({"id": r["id"], "w": w, "h": h, "n_holds": len(r["holds"])})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "gym_embeddings.npz", **store)
    json.dump(meta, open(OUT / "gym_meta.json", "w"))
    print("saved gym embeddings for", len(routes), "routes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="both", choices=["kilter", "gym", "both"])
    ap.add_argument("--k-views", type=int, default=24)
    ap.add_argument("--n-aug", type=int, default=12)
    ap.add_argument("--gym", default=str(ROOT.parent.parent / "gymdata"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.what in ("kilter", "both"):
        build_kilter_views(dev, args.k_views, args.seed)
    if args.what in ("gym", "both"):
        build_gym_embeddings(dev, args.gym, args.n_aug, args.seed)
