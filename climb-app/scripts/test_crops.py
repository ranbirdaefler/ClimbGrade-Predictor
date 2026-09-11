"""Test hold cropping + background masking on the gym dataset.

Downloads submissions from HuggingFace, runs the crop pipeline,
and saves per-route visualizations showing the masked 224x224 crops
with their mask method labels.

Usage: python scripts/test_crops.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.crop import crop_hold

_MASK_COLORS = {
    "color": "#22cc44",
    "grabcut": "#ff8800",
    "radial": "#cc3333",
}


def main() -> None:
    dataset_path = snapshot_download(
        repo_id="ranbirr1/climb-route-w-grade",
        repo_type="dataset",
        local_dir="data/gym_dataset",
    )

    submissions_dir = Path(dataset_path) / "submissions"
    output_dir = Path("results/crop_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/inference.yaml") as f:
        config = yaml.safe_load(f)

    all_mask_methods: list[str] = []
    all_results: list[dict] = []

    role_color_map = {
        "start": "lime", "finish": "red", "middle": "cyan",
        "foot": "yellow", "hand": "cyan", "volume": "magenta",
    }

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

        print(f"\n=== {meta['id']} ===")
        print(f"  Image: {img_w}x{img_h}")
        print(f"  Holds: {len(meta['holds'])}")
        print(f"  Actual: {meta.get('actual_grade', '?')}  "
              f"Predicted: {meta.get('predicted_grade', '?')}")

        # -- Overview image with tap points + bboxes --
        fig_ov, ax_ov = plt.subplots(1, 1, figsize=(8, 8))
        ax_ov.imshow(image_np)
        ax_ov.set_title(
            f"Route {meta['id'][:8]}... | "
            f"Actual: {meta.get('actual_grade', '?')} | "
            f"Predicted: {meta.get('predicted_grade', '?')}"
        )

        crops: list[Image.Image] = []
        debugs: list[dict] = []

        for i, hold in enumerate(meta["holds"]):
            tx, ty = hold["tap_x"], hold["tap_y"]
            role = hold["role"]

            crop_pil, bbox, debug = crop_hold(image_np, tx, ty, config)
            crops.append(crop_pil)
            debugs.append(debug)
            all_mask_methods.append(debug["mask_method"])

            color = role_color_map.get(role, "cyan")
            ax_ov.plot(
                tx, ty, "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5,
            )
            ax_ov.annotate(
                str(i + 1), (tx + 8, ty - 8),
                color="white", fontsize=9, fontweight="bold",
            )
            if bbox:
                bx1, by1, bx2, by2 = bbox
                rect = plt.Rectangle(
                    (bx1, by1), bx2 - bx1, by2 - by1,
                    linewidth=1.5, edgecolor=color,
                    facecolor="none", linestyle="--",
                )
                ax_ov.add_patch(rect)

            print(
                f"  Hold {i + 1}: ({tx},{ty}) role={role} "
                f"mask={debug['mask_method']} radius={debug['crop_radius']}"
            )

        ax_ov.axis("off")
        fig_ov.tight_layout()
        fig_ov.savefig(output_dir / f"{meta['id']}_overview.png", dpi=100, bbox_inches="tight")
        plt.close(fig_ov)

        # -- Crop grid with mask method labels --
        n_crops = len(crops)
        cols = min(4, n_crops)
        rows = (n_crops + cols - 1) // cols

        fig_cr, axes_cr = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        if rows == 1 and cols == 1:
            axes_cr = np.array([[axes_cr]])
        elif rows == 1:
            axes_cr = axes_cr[np.newaxis, :]
        elif cols == 1:
            axes_cr = axes_cr[:, np.newaxis]

        for i, (crop, dbg) in enumerate(zip(crops, debugs)):
            r, c = i // cols, i % cols
            axes_cr[r, c].imshow(crop)
            role = meta["holds"][i]["role"]
            mm = dbg["mask_method"]
            title_color = _MASK_COLORS.get(mm, "black")
            axes_cr[r, c].set_title(
                f"Hold {i + 1} ({role}) [{mm}]",
                fontsize=7, color=title_color, fontweight="bold",
            )
            axes_cr[r, c].axis("off")

        for i in range(n_crops, rows * cols):
            r, c = i // cols, i % cols
            axes_cr[r, c].axis("off")

        fig_cr.tight_layout()
        fig_cr.savefig(output_dir / f"{meta['id']}_crops.png", dpi=100, bbox_inches="tight")
        plt.close(fig_cr)

        all_results.append({
            "id": meta["id"],
            "n_holds": n_crops,
            "image_size": f"{img_w}x{img_h}",
            "actual": meta.get("actual_grade"),
            "predicted": meta.get("predicted_grade"),
            "mask_methods": [d["mask_method"] for d in debugs],
        })

    # -- Summary --
    print("\n" + "=" * 60)
    print("CROP + MASK TEST SUMMARY")
    print("=" * 60)

    total = len(all_mask_methods)
    if total > 0:
        color_n = all_mask_methods.count("color")
        gc_n = all_mask_methods.count("grabcut")
        rad_n = all_mask_methods.count("radial")
        print(f"Routes tested:   {len(all_results)}")
        print(f"Total holds:     {total}")
        print(f"  Color mask:    {color_n:>3}/{total} ({100 * color_n / total:.0f}%)")
        print(f"  GrabCut mask:  {gc_n:>3}/{total} ({100 * gc_n / total:.0f}%)")
        print(f"  Radial fade:   {rad_n:>3}/{total} ({100 * rad_n / total:.0f}%)")
    else:
        print("No submissions with images found.")

    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
