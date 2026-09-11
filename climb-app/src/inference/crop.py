"""Smart hold cropping using color-based segmentation in LAB space."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage import measure


def crop_hold(
    image: np.ndarray,
    tap_x: int,
    tap_y: int,
    config: dict,
) -> tuple[Image.Image, np.ndarray | None, tuple[int, int, int, int]]:
    """Crop a hold using color segmentation around the tap point.

    Returns (crop_pil_224x224, hold_mask_or_None, bbox).
    """
    cfg = config["crop"]
    h, w = image.shape[:2]
    tap_x, tap_y = int(np.clip(tap_x, 0, w - 1)), int(np.clip(tap_y, 0, h - 1))

    sample_r = 5
    patch = image[
        max(0, tap_y - sample_r) : min(h, tap_y + sample_r + 1),
        max(0, tap_x - sample_r) : min(w, tap_x + sample_r + 1),
    ]
    hold_rgb = np.median(patch.reshape(-1, 3), axis=0)

    hold_lab = cv2.cvtColor(
        np.uint8([[hold_rgb]]), cv2.COLOR_RGB2LAB
    )[0, 0].astype(np.float32)
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    color_diff = np.sqrt(np.sum((image_lab - hold_lab) ** 2, axis=2))

    threshold = cfg["color_distance_threshold"]
    search_r = cfg["max_crop_size"] // 2
    rx1, ry1 = max(0, tap_x - search_r), max(0, tap_y - search_r)
    rx2, ry2 = min(w, tap_x + search_r), min(h, tap_y + search_r)

    tap_label = 0
    hold_mask = None
    for mult in (1.0, 1.5, 2.0, 2.5):
        mask = (color_diff < threshold * mult).astype(np.uint8)
        restricted = np.zeros_like(mask)
        restricted[ry1:ry2, rx1:rx2] = mask[ry1:ry2, rx1:rx2]
        labels = measure.label(restricted, connectivity=2)
        tap_label = labels[tap_y, tap_x]
        if tap_label > 0:
            hold_mask = (labels == tap_label).astype(np.uint8)
            break

    if tap_label == 0 or hold_mask is None:
        return _crop_fixed(image, tap_x, tap_y, cfg)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hold_mask = cv2.morphologyEx(hold_mask, cv2.MORPH_CLOSE, kernel)
    hold_mask = cv2.morphologyEx(hold_mask, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(hold_mask > 0)
    if len(xs) == 0:
        return _crop_fixed(image, tap_x, tap_y, cfg)

    bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = bx2 - bx1, by2 - by1

    pad = cfg["padding_fraction"]
    cx1 = max(0, bx1 - int(bw * pad))
    cy1 = max(0, by1 - int(bh * pad))
    cx2 = min(w, bx2 + int(bw * pad))
    cy2 = min(h, by2 + int(bh * pad))

    side = max(cx2 - cx1, cy2 - cy1)
    side = max(cfg["min_crop_size"], min(cfg["max_crop_size"], side))

    mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
    sx1 = max(0, mx - side // 2)
    sy1 = max(0, my - side // 2)
    sx2 = min(w, sx1 + side)
    sy2 = min(h, sy1 + side)
    if sx2 - sx1 < side:
        sx1 = max(0, sx2 - side)
    if sy2 - sy1 < side:
        sy1 = max(0, sy2 - side)

    crop = image[sy1:sy2, sx1:sx2]
    crop_pil = Image.fromarray(crop).resize(
        (cfg["output_size"], cfg["output_size"]), Image.LANCZOS
    )
    return crop_pil, hold_mask, (bx1, by1, bx2, by2)


crop_hold_smart = crop_hold


def _crop_fixed(
    image: np.ndarray, tap_x: int, tap_y: int, cfg: dict
) -> tuple[Image.Image, None, tuple[int, int, int, int]]:
    """Fallback: fixed-size square crop centred on the tap point."""
    h, w = image.shape[:2]
    r = cfg["base_radius"]
    x1, y1 = max(0, tap_x - r), max(0, tap_y - r)
    x2, y2 = min(w, tap_x + r), min(h, tap_y + r)
    crop = image[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop).resize(
        (cfg["output_size"], cfg["output_size"]), Image.LANCZOS
    )
    return crop_pil, None, (x1, y1, x2, y2)
