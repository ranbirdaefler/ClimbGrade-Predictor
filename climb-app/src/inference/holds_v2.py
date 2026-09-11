"""Hold cropping for gym photos, with optional Kilter-style preprocessing.

Mirrors the deployed ``crop_hold`` (LAB colour flood around the tap) but also
returns the hold mask inside the crop window so the background can be
replaced with neutral grey, the way the Kilter training crops were made.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage import measure

NEUTRAL_GREY = 128

CROP_CFG = {
    "base_radius": 40,
    "min_crop_size": 64,
    "max_crop_size": 400,
    "output_size": 224,
    "color_distance_threshold": 40,
    "expansion_steps": 5,
    "expansion_pixels": 8,
    "padding_fraction": 0.25,
}


def segment_hold(image: np.ndarray, tap_x: int, tap_y: int, cfg: dict = CROP_CFG) -> np.ndarray | None:
    """Colour-flood mask (full image size) of the hold under the tap, or None."""
    h, w = image.shape[:2]
    tap_x, tap_y = int(np.clip(tap_x, 0, w - 1)), int(np.clip(tap_y, 0, h - 1))
    r = 5
    patch = image[max(0, tap_y - r): min(h, tap_y + r + 1), max(0, tap_x - r): min(w, tap_x + r + 1)]
    hold_rgb = np.median(patch.reshape(-1, 3), axis=0)
    hold_lab = cv2.cvtColor(np.uint8([[hold_rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    color_diff = np.sqrt(np.sum((image_lab - hold_lab) ** 2, axis=2))

    search_r = cfg["max_crop_size"] // 2
    rx1, ry1 = max(0, tap_x - search_r), max(0, tap_y - search_r)
    rx2, ry2 = min(w, tap_x + search_r), min(h, tap_y + search_r)
    for mult in (1.0, 1.5, 2.0, 2.5):
        mask = (color_diff < cfg["color_distance_threshold"] * mult).astype(np.uint8)
        restricted = np.zeros_like(mask)
        restricted[ry1:ry2, rx1:rx2] = mask[ry1:ry2, rx1:rx2]
        labels = measure.label(restricted, connectivity=2)
        lab = labels[tap_y, tap_x]
        if lab > 0:
            hold_mask = (labels == lab).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            hold_mask = cv2.morphologyEx(hold_mask, cv2.MORPH_CLOSE, kernel)
            hold_mask = cv2.morphologyEx(hold_mask, cv2.MORPH_OPEN, kernel)
            if hold_mask.sum() == 0:
                return None
            # A flood that escaped onto the wall is useless as a hold mask.
            ys, xs = np.where(hold_mask > 0)
            if (xs.max() - xs.min()) > 0.3 * w or (ys.max() - ys.min()) > 0.3 * h:
                return None
            return hold_mask
    return None


def crop_window(image_shape: tuple, tap_x: int, tap_y: int, mask: np.ndarray | None, cfg: dict = CROP_CFG) -> tuple[int, int, int, int]:
    """Square crop window (x1, y1, x2, y2) around the hold."""
    h, w = image_shape[:2]
    if mask is None:
        r = cfg["base_radius"]
        return max(0, tap_x - r), max(0, tap_y - r), min(w, tap_x + r), min(h, tap_y + r)
    ys, xs = np.where(mask > 0)
    bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = bx2 - bx1, by2 - by1
    pad = cfg["padding_fraction"]
    cx1, cy1 = max(0, bx1 - int(bw * pad)), max(0, by1 - int(bh * pad))
    cx2, cy2 = min(w, bx2 + int(bw * pad)), min(h, by2 + int(bh * pad))
    side = max(cx2 - cx1, cy2 - cy1)
    side = max(cfg["min_crop_size"], min(cfg["max_crop_size"], side))
    mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
    sx1, sy1 = max(0, mx - side // 2), max(0, my - side // 2)
    sx2, sy2 = min(w, sx1 + side), min(h, sy1 + side)
    if sx2 - sx1 < side:
        sx1 = max(0, sx2 - side)
    if sy2 - sy1 < side:
        sy1 = max(0, sy2 - side)
    return sx1, sy1, sx2, sy2


def crop_hold(
    image: np.ndarray,
    tap_x: int,
    tap_y: int,
    mode: str = "raw",
    cfg: dict = CROP_CFG,
    jitter: tuple[int, int] = (0, 0),
) -> tuple[Image.Image, bool]:
    """Crop the hold at (tap_x, tap_y).

    mode: "raw"       - crop as deployed today
          "mask"      - background outside the colour mask -> neutral grey
          "gray"      - crop converted to greyscale (3-channel)
          "gray_mask" - both
    jitter: shift applied to the crop window (augmentation), not to the tap.
    Returns (crop_pil_224, mask_found).
    """
    h, w = image.shape[:2]
    tap_x, tap_y = int(np.clip(tap_x, 0, w - 1)), int(np.clip(tap_y, 0, h - 1))
    mask = segment_hold(image, tap_x, tap_y, cfg)
    x1, y1, x2, y2 = crop_window(image.shape, tap_x, tap_y, mask, cfg)
    if jitter != (0, 0):
        jx, jy = jitter
        side_x, side_y = x2 - x1, y2 - y1
        x1 = int(np.clip(x1 + jx, 0, w - side_x)); x2 = x1 + side_x
        y1 = int(np.clip(y1 + jy, 0, h - side_y)); y2 = y1 + side_y
    crop = image[y1:y2, x1:x2].copy()
    if "mask" in mode:
        if mask is not None:
            m = mask[y1:y2, x1:x2].astype(np.float32)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            m = cv2.dilate(m, kernel, iterations=1)[:, :, None]
            grey = np.full_like(crop, NEUTRAL_GREY)
            crop = np.clip(crop.astype(np.float32) * m + grey.astype(np.float32) * (1 - m), 0, 255).astype(np.uint8)
    if "gray" in mode:
        g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crop = np.stack([g, g, g], axis=-1)
    out = Image.fromarray(crop).resize((cfg["output_size"], cfg["output_size"]), Image.LANCZOS)
    return out, mask is not None
