"""Load the real-gym feedback dataset (HF: ranbirr1/climb-route-w-grade).

Each submission folder has metadata.json (holds, wall_angle, predicted and
optional actual grade) and wall_photo.jpg. Only submissions with an
``actual_grade`` are usable as labels. Duplicate photos (same image bytes)
are collapsed so that a route never appears in both a train and a test fold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

V_TO_DIFF = {  # V-grade -> Kilter difficulty (centre of the bin)
    "V0": 11, "V1": 13.5, "V2": 15, "V3": 16.5, "V4": 18.5, "V5": 20.5,
    "V6": 22, "V7": 23, "V8": 24.5, "V9": 26, "V10": 27, "V11": 28,
    "V12": 29, "V13": 30,
}
_GRADE_TABLE = [
    (10, "V0"), (11, "V0"), (12, "V0"), (13, "V1"), (14, "V1"), (15, "V2"),
    (16, "V3"), (17, "V3"), (18, "V4"), (19, "V4"), (20, "V5"), (21, "V5"),
    (22, "V6"), (23, "V7"), (24, "V8"), (25, "V8"), (26, "V9"), (27, "V10"),
    (28, "V11"), (29, "V12"), (30, "V13"),
]


def diff_to_v(diff: float) -> str:
    diff = max(10.0, min(30.0, diff))
    for threshold, grade in _GRADE_TABLE:
        if diff <= threshold + 0.5:
            return grade
    return "V13"


def v_index(g: str) -> int:
    return int(str(g).replace("V", "").replace("+", ""))


def load_gym_routes(root: str | Path, labeled_only: bool = True) -> list[dict]:
    root = Path(root)
    routes = []
    seen_hashes: dict[str, dict] = {}
    for meta_path in sorted((root / "submissions").glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        img_path = meta_path.parent / meta.get("image_filename", "wall_photo.jpg")
        if not img_path.exists():
            continue
        actual = meta.get("actual_grade")
        if labeled_only and not actual:
            continue
        holds = [h for h in meta["holds"] if "tap_x" in h]
        if len(holds) < 3:
            continue
        digest = hashlib.md5(img_path.read_bytes()).hexdigest()
        key = digest + "|" + repr(sorted((h["tap_x"], h["tap_y"]) for h in holds))
        route = {
            "id": meta["id"],
            "image_path": str(img_path),
            "image_hash": digest,
            "holds": holds,
            "wall_angle": float(meta.get("wall_angle", 30)),
            "actual_grade": actual,
            "actual_diff": V_TO_DIFF[actual] if actual else None,
            "actual_v": v_index(actual) if actual else None,
            "old_predicted_grade": meta.get("predicted_grade"),
            "old_predicted_diff": meta.get("predicted_difficulty"),
            "timestamp": meta.get("timestamp", ""),
        }
        # Exact duplicate (same photo + same taps): keep the labeled one.
        if key in seen_hashes:
            if actual and not seen_hashes[key]["actual_grade"]:
                seen_hashes[key] = route
            continue
        seen_hashes[key] = route
    routes = list(seen_hashes.values())
    routes.sort(key=lambda r: r["id"])
    return routes


def load_image(route: dict) -> np.ndarray:
    from PIL import ImageOps

    img = Image.open(route["image_path"])
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.asarray(img)


def summarize(routes: list[dict]) -> str:
    import collections

    grades = collections.Counter(r["actual_grade"] for r in routes)
    groups = len({r["image_hash"] for r in routes})
    return f"{len(routes)} labeled routes, {groups} distinct photos, grades: {dict(sorted(grades.items(), key=lambda kv: v_index(kv[0])))}"


if __name__ == "__main__":
    import sys

    rs = load_gym_routes(sys.argv[1] if len(sys.argv) > 1 else "../../gymdata")
    print(summarize(rs))
