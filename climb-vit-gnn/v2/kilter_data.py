"""Kilter Board routes + hold crops for v2 pre-training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

HOLD_TYPES = ["jug", "crimp", "sloper", "pinch", "foot"]
TYPE_TO_IDX = {t: i for i, t in enumerate(HOLD_TYPES)}


def load_routes(split: str) -> list[dict]:
    return json.load(open(DATA / "processed" / f"{split}.json"))


def load_crop_index() -> tuple[dict[int, int], list[int]]:
    """placement_id -> hole_id, and the sorted list of hole ids with crops."""
    mapping = {int(k): int(v) for k, v in json.load(open(DATA / "hold_crops" / "placement_to_crop.json")).items()}
    hole_ids = sorted({v for v in mapping.values() if (DATA / "hold_crops" / f"hold_{v}.png").exists()})
    return mapping, hole_ids


def load_crop(hole_id: int) -> Image.Image:
    return Image.open(DATA / "hold_crops" / f"hold_{hole_id}.png").convert("RGB")


def hold_attribute_table(routes: list[dict]) -> dict[int, dict]:
    """placement_id -> {type, size, depth, orientation} (constant per placement)."""
    table: dict[int, dict] = {}
    for r in routes:
        for h in r["holds"]:
            pid = h["placement_id"]
            if pid not in table:
                table[pid] = {
                    "type": h.get("type", "jug"),
                    "size": int(h.get("size", 3)),
                    "depth": int(h.get("depth", 2)),
                    "orientation": float(h.get("orientation", 0)),
                }
    return table


def crop_background_mask(crop: np.ndarray) -> np.ndarray:
    """1 where the Kilter crop is the neutral-grey masked background."""
    import cv2

    bg = np.all(np.abs(crop.astype(int) - 128) <= 1, axis=-1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel)
    return bg
