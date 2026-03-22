"""SAM2-tiny masking experiment.

End-to-end pipeline:
  1. Load SAM2-tiny and mask all Kilter Board hold crops
  2. Recompute DINOv2 embeddings on the SAM2-masked crops
  3. Retrain the GNN on the new embeddings
  4. Test on gym dataset and compare against old model predictions

Usage:
    cd climb-vit-gnn
    python experiments/sam2_masking/run_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP_DIR = Path(__file__).resolve().parent
CROPS_DIR = EXP_DIR / "sam2_crops"
EMBED_DIR = EXP_DIR / "sam2_embeddings"
RESULTS_DIR = EXP_DIR / "results"

_NEUTRAL_GREY = 128
SAM2_HF_MODEL_ID = "facebook/sam2.1-hiera-tiny"


# ── Step 1: SAM2 masking on Kilter Board crops ──────────────────────

def step1_sam2_mask_crops() -> None:
    """Mask all Kilter Board hold crops using SAM2-tiny point prompts."""
    print("\n" + "=" * 60)
    print("STEP 1: SAM2 masking on Kilter Board hold crops")
    print("=" * 60)

    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Loading SAM2-tiny from HuggingFace ({SAM2_HF_MODEL_ID})...")

    sam2_model = build_sam2_hf(SAM2_HF_MODEL_ID, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    original_crops_dir = ROOT / "data" / "hold_crops"
    crop_files = sorted(original_crops_dir.glob("hold_*.png"))
    print(f"  Found {len(crop_files)} hold crops")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"sam2_ok": 0, "sam2_fail": 0}

    for crop_path in tqdm(crop_files, desc="  SAM2 masking"):
        img = Image.open(crop_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        predictor.set_image(img_np)

        # Point prompt: center of the image (where the hold is)
        point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Pick the mask with the highest confidence
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.float32)

        fg_fraction = mask.sum() / (h * w)
        if fg_fraction < 0.03 or fg_fraction > 0.95:
            stats["sam2_fail"] += 1
            # Fallback: just replace very dark pixels with grey
            mask = _dark_threshold_mask(img_np)
        else:
            stats["sam2_ok"] += 1
            # Slight dilation to avoid clipping hold edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=1)

        masked = _apply_mask(img_np, mask)
        Image.fromarray(masked).save(CROPS_DIR / crop_path.name)

    # Copy placement_to_crop.json (same mapping, different crops)
    src_mapping = original_crops_dir / "placement_to_crop.json"
    if src_mapping.exists():
        import shutil
        shutil.copy(src_mapping, CROPS_DIR / "placement_to_crop.json")

    total = stats["sam2_ok"] + stats["sam2_fail"]
    print(f"\n  SAM2 succeeded: {stats['sam2_ok']}/{total} ({100 * stats['sam2_ok'] / total:.0f}%)")
    print(f"  Fell back to threshold: {stats['sam2_fail']}/{total}")


def _dark_threshold_mask(img_np: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    return (lightness > 40).astype(np.float32)


def _apply_mask(region: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grey = np.full_like(region, _NEUTRAL_GREY)
    mask_3ch = mask[:, :, np.newaxis]
    blended = (region.astype(np.float32) * mask_3ch
               + grey.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── Step 2: DINOv2 embeddings ───────────────────────────────────────

def step2_embed() -> None:
    """Compute DINOv2 embeddings on SAM2-masked crops."""
    print("\n" + "=" * 60)
    print("STEP 2: DINOv2 embeddings on SAM2-masked crops")
    print("=" * 60)

    from src.data.embed_holds import embed_all_holds, save_embeddings

    config = yaml.safe_load(open(ROOT / "configs" / "vit_gnn.yaml"))
    vit_cfg = config["vit"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings, hold_ids = embed_all_holds(
        str(CROPS_DIR),
        model_name=vit_cfg["model_name"],
        device=device,
        batch_size=vit_cfg["batch_size"],
    )

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeddings, hold_ids, str(EMBED_DIR))


# ── Step 3: Retrain GNN ────────────────────────────────────────────

def step3_retrain() -> dict:
    """Retrain the GNN on SAM2-masked embeddings. Returns history."""
    print("\n" + "=" * 60)
    print("STEP 3: Retrain GNN on SAM2-masked embeddings")
    print("=" * 60)

    import random

    from torch_geometric.loader import DataLoader

    from src.data.dataset_vit import KilterViTDataset
    from src.data.embed_holds import load_embeddings
    from src.models.gnn_vit import KilterViTGNN
    from src.train_vit import train_vit_gnn

    config = yaml.safe_load(open(ROOT / "configs" / "retrain_full.yaml"))
    data_cfg = config["data"]
    train_cfg = config["training"]

    emb_matrix, crop_id_to_idx = load_embeddings(str(EMBED_DIR))
    print(f"  Embeddings: {emb_matrix.shape}")

    with open(CROPS_DIR / "placement_to_crop.json") as f:
        placement_to_crop = {int(k): int(v) for k, v in json.load(f).items()}
    print(f"  Placement -> crop: {len(placement_to_crop)} entries")

    all_routes = json.load(open(data_cfg["routes_path"]))
    print(f"  Total routes: {len(all_routes)}")

    uuids = list({r["uuid"] for r in all_routes})
    random.seed(data_cfg["random_seed"])
    random.shuffle(uuids)
    val_count = int(len(uuids) * data_cfg["val_size"])
    val_uuids = set(uuids[:val_count])
    train_uuids = set(uuids[val_count:])

    train_routes = [r for r in all_routes if r["uuid"] in train_uuids]
    val_routes = [r for r in all_routes if r["uuid"] in val_uuids]
    print(f"  Train: {len(train_routes)}  Val: {len(val_routes)}")

    train_ds = KilterViTDataset(train_routes, emb_matrix, placement_to_crop, crop_id_to_idx, foot_drop_prob=0.8)
    val_ds = KilterViTDataset(val_routes, emb_matrix, placement_to_crop, crop_id_to_idx)
    print(f"  Graphs: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = KilterViTGNN(config).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n  Training...")
    history = train_vit_gnn(model, train_loader, val_loader, config, device, save_dir=str(RESULTS_DIR))

    # Save experiment checkpoint
    best_val_mae = min(history["val_mae"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "crop_id_to_idx": crop_id_to_idx,
            "placement_to_crop": placement_to_crop,
            "training_stats": {
                "train_routes": len(train_ds),
                "val_routes": len(val_ds),
                "best_val_mae": best_val_mae,
                "epochs_trained": len(history["train_loss"]),
            },
        },
        RESULTS_DIR / "sam2_model.pt",
    )
    print(f"\n  Saved sam2_model.pt (best val MAE: {best_val_mae:.4f})")

    return history


# ── Step 4: Test on gym dataset ─────────────────────────────────────

def step4_test_on_gym() -> None:
    """Run inference on the collected gym dataset and compare."""
    print("\n" + "=" * 60)
    print("STEP 4: Test on gym dataset")
    print("=" * 60)

    import math

    from huggingface_hub import snapshot_download
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader as PyGLoader
    from torchvision import transforms as T

    from src.models.gnn_vit import KilterViTGNN

    dataset_path = snapshot_download(
        repo_id="ranbirr1/climb-route-w-grade",
        repo_type="dataset",
        local_dir=str(ROOT / "data" / "gym_dataset"),
    )
    submissions_dir = Path(dataset_path) / "submissions"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SAM2 for gym crop masking
    print("  Loading SAM2-tiny for gym crop masking...")
    sam2_model = build_sam2_hf(SAM2_HF_MODEL_ID, device=str(device))
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Load the experiment model
    print("  Loading SAM2-trained GNN model...")
    checkpoint = torch.load(RESULTS_DIR / "sam2_model.pt", map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    model = KilterViTGNN(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Inline DINOv2 embedder (avoids dependency on climb-app's src.inference)
    print("  Loading DINOv2 embedder...")
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dino_model = dino_model.to(device).eval()
    dino_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def embed_crops(crops: list) -> torch.Tensor:
        tensors = torch.stack([dino_transform(c) for c in crops]).to(device)
        return dino_model(tensors)

    def build_graph(hold_data, wall_angle, image_shape):
        n = len(hold_data)
        img_h, img_w = image_shape[:2]
        vit_embs = torch.stack([h["embedding"] for h in hold_data])
        role_map = {"start": 0, "hand": 1, "middle": 1, "finish": 2, "foot": 3, "foot_only": 3, "volume": 1}
        role_idx = torch.tensor([role_map.get(h["role"], 1) for h in hold_data], dtype=torch.long)
        positions = torch.tensor(
            [[h["tap_x"] / img_w, 1.0 - h["tap_y"] / img_h] for h in hold_data], dtype=torch.float,
        )
        wall_angle_norm = torch.tensor([wall_angle / 70.0], dtype=torch.float)
        src, dst, edge_attrs = [], [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
                    dx = positions[j][0].item() - positions[i][0].item()
                    dy = positions[j][1].item() - positions[i][1].item()
                    dist = math.sqrt(dx * dx + dy * dy)
                    ang = math.atan2(dy, dx)
                    edge_attrs.append([dx, dy, dist, math.sin(ang), math.cos(ang)])
        return PyGData(
            vit_emb=vit_embs, role_idx=role_idx, pos_features=positions,
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float) if src else torch.zeros((0, 5)),
            wall_angle=wall_angle_norm, num_nodes=n,
        )

    def diff_to_grade(d):
        table = [
            (10, "V0"), (11, "V0"), (12, "V0"),
            (13, "V1"), (14, "V1"), (15, "V2"),
            (16, "V3"), (17, "V3"), (18, "V4"), (19, "V4"),
            (20, "V5"), (21, "V5"), (22, "V6"), (23, "V7"),
            (24, "V8"), (25, "V8"), (26, "V9"), (27, "V10"),
            (28, "V11"), (29, "V12"), (30, "V13"),
        ]
        d = max(10.0, min(30.0, d))
        for threshold, grade in table:
            if d <= threshold + 0.5:
                return grade
        return "V13+"

    results = []

    for sub_dir in sorted(submissions_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        metadata_path = sub_dir / "metadata.json"
        image_path = sub_dir / "wall_photo.jpg"
        if not metadata_path.exists() or not image_path.exists():
            continue

        with open(metadata_path) as f:
            meta = json.load(f)

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        img_h, img_w = image_np.shape[:2]

        # Crop each hold and apply SAM2 masking
        crops = []
        min_dim = min(img_h, img_w)
        crop_radius = int(min_dim * 0.10)
        crop_radius = max(24, min(400, crop_radius))

        for hold in meta["holds"]:
            tx, ty = int(hold["tap_x"]), int(hold["tap_y"])
            tx = max(0, min(img_w - 1, tx))
            ty = max(0, min(img_h - 1, ty))

            rx1 = max(0, tx - crop_radius)
            ry1 = max(0, ty - crop_radius)
            rx2 = min(img_w, tx + crop_radius)
            ry2 = min(img_h, ty + crop_radius)
            region = image_np[ry1:ry2, rx1:rx2].copy()
            local_tx = tx - rx1
            local_ty = ty - ry1

            # SAM2 point prompt on the region
            sam2_predictor.set_image(region)
            point_coords = np.array([[local_tx, local_ty]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

            masks, scores, _ = sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(np.float32)

            fg_frac = mask.sum() / (region.shape[0] * region.shape[1])
            if fg_frac < 0.03 or fg_frac > 0.95:
                mask = np.ones(region.shape[:2], dtype=np.float32)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.dilate(mask, kernel, iterations=1)

            masked = _apply_mask(region, mask)

            crop_pil = Image.fromarray(masked)
            cw, ch = crop_pil.size
            if cw != ch:
                side = max(cw, ch)
                sq = Image.new("RGB", (side, side), (_NEUTRAL_GREY,) * 3)
                sq.paste(crop_pil, ((side - cw) // 2, (side - ch) // 2))
                crop_pil = sq
            crop_pil = crop_pil.resize((224, 224), Image.LANCZOS)
            crops.append(crop_pil)

        # Embed and build graph
        embeddings = embed_crops(crops)
        hold_data = []
        for i, hold in enumerate(meta["holds"]):
            hold_data.append({
                "embedding": embeddings[i],
                "tap_x": hold["tap_x"],
                "tap_y": hold["tap_y"],
                "role": hold["role"],
            })

        data = build_graph(hold_data, meta["wall_angle"], image_np.shape)
        loader = PyGLoader([data], batch_size=1)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            prediction = model(batch).item()

        new_grade = diff_to_grade(prediction)
        actual = meta.get("actual_grade", "?")
        old_pred = meta.get("predicted_grade", "?")

        results.append({
            "id": meta["id"],
            "actual": actual,
            "old_predicted": old_pred,
            "sam2_predicted": new_grade,
            "sam2_difficulty": round(prediction, 1),
        })

    # Print comparison
    print(f"\n  {'ID':>12}  {'Actual':>8}  {'Old':>8}  {'SAM2':>8}  {'Diff':>6}")
    print("  " + "-" * 50)
    for r in results:
        print(
            f"  {r['id'][:12]:>12}  {r['actual']:>8}  "
            f"{r['old_predicted']:>8}  {r['sam2_predicted']:>8}  "
            f"{r['sam2_difficulty']:>6.1f}"
        )

    with open(RESULTS_DIR / "gym_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'gym_comparison.json'}")


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    print("SAM2-tiny Masking Experiment")
    print("=" * 60)

    step1_sam2_mask_crops()
    step2_embed()
    step3_retrain()
    step4_test_on_gym()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0, help="Run a single step (1-4), or 0 for all")
    args = parser.parse_args()

    if args.step == 0:
        main()
    else:
        {1: step1_sam2_mask_crops, 2: step2_embed, 3: step3_retrain, 4: step4_test_on_gym}[args.step]()
