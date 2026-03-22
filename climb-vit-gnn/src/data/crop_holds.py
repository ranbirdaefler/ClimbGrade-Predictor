"""Crop hold patches from Kilter Board layout images.

Uses two board sizes to achieve 100% hold coverage:
- 12x14 Commercial (id=7): covers most bolt-ons including tall y
- 16x12 Super Wide (id=28): covers screw-ons + bolt-ons with negative x

After cropping, the dark board background is replaced with neutral grey
(128, 128, 128) to match the inference-time masking in the deployed app.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_NEUTRAL_GREY = 128


def _load_db_info(db_path: str) -> dict:
    """Load all placement/hole/board info from the DB."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    boards = {}
    for psid in (7, 28):
        cur.execute(
            "SELECT edge_left, edge_right, edge_bottom, edge_top "
            "FROM product_sizes WHERE id = ?",
            (psid,),
        )
        r = cur.fetchone()
        boards[psid] = {
            "edge_left": r[0], "edge_right": r[1],
            "edge_bottom": r[2], "edge_top": r[3],
        }
        cur.execute(
            "SELECT set_id, image_filename "
            "FROM product_sizes_layouts_sets "
            "WHERE product_size_id = ? AND layout_id = 1",
            (psid,),
        )
        boards[psid]["images"] = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute(
        "SELECT p.id, p.hole_id, p.set_id, h.x, h.y "
        "FROM placements p JOIN holes h ON h.id = p.hole_id "
        "WHERE p.layout_id = 1"
    )
    placements = [
        {"placement_id": r[0], "hole_id": r[1], "set_id": r[2], "x": r[3], "y": r[4]}
        for r in cur.fetchall()
    ]
    conn.close()
    return {"boards": boards, "placements": placements}


def _fits_board(x: int, y: int, edges: dict) -> bool:
    return (
        edges["edge_left"] <= x <= edges["edge_right"]
        and edges["edge_bottom"] <= y <= edges["edge_top"]
    )


def _coord_to_pixel(
    hx: int, hy: int, edges: dict, img_w: int, img_h: int
) -> tuple[int, int]:
    bw = edges["edge_right"] - edges["edge_left"]
    bh = edges["edge_top"] - edges["edge_bottom"]
    fx = (hx - edges["edge_left"]) / bw
    fy = (hy - edges["edge_bottom"]) / bh
    return int(fx * img_w), int((1.0 - fy) * img_h)


def _mask_background(crop_pil: Image.Image) -> Image.Image:
    """Replace the dark board background with neutral grey.

    The Kilter Board renders have holds on a near-black background.
    We threshold in LAB lightness to find the background, then use
    GrabCut (seeded with that mask) to refine the boundary.
    """
    crop_np = np.array(crop_pil)
    h, w = crop_np.shape[:2]

    lab = cv2.cvtColor(crop_np, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)

    # Dark pixels (L < 40 out of 255) are likely background
    bg_thresh = 40
    dark_mask = (lightness < bg_thresh).astype(np.uint8)

    # Use GrabCut to refine: dark = definite BG, bright = probable FG
    gc_mask = np.where(dark_mask, cv2.GC_BGD, cv2.GC_PR_FGD).astype(np.uint8)

    # Mark the center region as definite foreground
    cx, cy = w // 2, h // 2
    cr = max(3, min(h, w) // 8)
    gc_mask[
        max(0, cy - cr):min(h, cy + cr),
        max(0, cx - cr):min(w, cx + cr),
    ] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(
            crop_bgr, gc_mask, None,
            bgd_model, fgd_model,
            iterCount=3, mode=cv2.GC_INIT_WITH_MASK,
        )
        fg = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.float32)
    except cv2.error:
        fg = (1.0 - dark_mask).astype(np.float32)

    # Slight dilation so we don't clip hold edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.dilate(fg, kernel, iterations=1)

    grey_bg = np.full_like(crop_np, _NEUTRAL_GREY)
    mask_3ch = fg[:, :, np.newaxis]
    result = (crop_np.astype(np.float32) * mask_3ch
              + grey_bg.astype(np.float32) * (1.0 - mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def crop_all_holds(
    db_path: str,
    board_images_dir: str | Path,
    crop_size: int = 224,
    padding_factor: float = 1.5,
    output_dir: str | Path = "data/hold_crops",
) -> dict[int, str]:
    """Crop a patch around every hold. Returns hole_id -> crop path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    board_images_dir = Path(board_images_dir)

    info = _load_db_info(db_path)
    boards = info["boards"]

    loaded: dict[tuple[int, int], tuple[Image.Image, dict]] = {}
    for psid, bdata in boards.items():
        for set_id, img_rel in bdata["images"].items():
            img_path = board_images_dir / img_rel
            if not img_path.exists():
                img_path = board_images_dir / Path(img_rel).name
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                loaded[(psid, set_id)] = (img, bdata)
                print(f"  Loaded ps={psid} set={set_id}: {img_path.name} {img.size}")

    crops: dict[int, str] = {}
    skipped = 0

    for pl in info["placements"]:
        hid, sid, hx, hy = pl["hole_id"], pl["set_id"], pl["x"], pl["y"]

        # Strategy: prefer 12x14 (id=7) for bolt-ons, 16x12 (id=28) for screw-ons
        # Fall back to the other board if the hold doesn't fit
        if sid == 1:
            candidates = [(7, sid), (28, sid)]
        else:
            candidates = [(28, sid), (7, sid)]

        cropped = False
        for psid, s in candidates:
            key = (psid, s)
            if key not in loaded:
                continue
            img, bdata = loaded[key]
            if not _fits_board(hx, hy, bdata):
                continue

            img_w, img_h = img.size
            bw = bdata["edge_right"] - bdata["edge_left"]
            bh = bdata["edge_top"] - bdata["edge_bottom"]
            grid_spacing = min(img_w / (bw / 4), img_h / (bh / 4))
            radius = int(grid_spacing * padding_factor / 2)

            px, py = _coord_to_pixel(hx, hy, bdata, img_w, img_h)
            left = max(0, px - radius)
            top = max(0, py - radius)
            right = min(img_w, px + radius)
            bottom = min(img_h, py + radius)

            if right <= left or bottom <= top or (right - left) < 5:
                continue

            crop = img.crop((left, top, right, bottom))
            crop = crop.resize((crop_size, crop_size), Image.LANCZOS)
            crop = _mask_background(crop)
            path = output_dir / f"hold_{hid}.png"
            crop.save(path)
            crops[hid] = str(path)
            cropped = True
            break

        if not cropped:
            skipped += 1

    print(f"  Cropped {len(crops)} holds, skipped {skipped}")
    return crops


def build_placement_to_crop_mapping(
    db_path: str,
    crops: dict[int, str],
) -> dict[int, int]:
    """Map placement_id -> hole_id for holds that have crops."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, hole_id FROM placements WHERE layout_id = 1")
    mapping = {}
    for pid, hid in cur.fetchall():
        if hid in crops:
            mapping[pid] = hid
    conn.close()
    print(f"  Placement -> hole_id mapping: {len(mapping)} entries")
    return mapping
