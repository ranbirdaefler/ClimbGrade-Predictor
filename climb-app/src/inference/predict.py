"""Full inference pipeline: image + tap points -> grade prediction.

Two model generations are supported:
  * v2 (default): ClimbGNNv2 ensemble pre-trained on Kilter Board routes and
    fine-tuned on real gym routes; grey-masked hold crops; distributional
    output with a calibrated grade range.
  * v1 (legacy): single KilterViTGNN with raw crops and a fixed +-1 range.
Which one runs is decided by ``model.checkpoint`` in configs/inference.yaml.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .crop import crop_hold as crop_hold_v1
from .embed import HoldEmbedder

# Kilter Board difficulty -> V-grade (from difficulty_grades table)
_GRADE_TABLE = [
    (10, "V0"), (11, "V0"), (12, "V0"),
    (13, "V1"), (14, "V1"),
    (15, "V2"),
    (16, "V3"), (17, "V3"),
    (18, "V4"), (19, "V4"),
    (20, "V5"), (21, "V5"),
    (22, "V6"),
    (23, "V7"),
    (24, "V8"), (25, "V8"),
    (26, "V9"),
    (27, "V10"),
    (28, "V11"),
    (29, "V12"),
    (30, "V13"),
]


def _difficulty_to_vgrade(diff: float) -> str:
    diff = max(10.0, min(30.0, diff))
    for threshold, grade in _GRADE_TABLE:
        if diff <= threshold + 0.5:
            return grade
    return "V13+"


class GradePredictor:
    """Load model + embedder once; call ``predict`` for each route."""

    def __init__(self, config: dict):
        self.config = config
        device_str = config["model"]["device"]
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        ckpt_path = config["model"]["checkpoint"]
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.version = 2 if "members" in checkpoint else 1

        if self.version == 2:
            from .v2 import EnsembleV2

            self.ensemble = EnsembleV2(ckpt_path, device=device_str)
            self.crop_mode = self.ensemble.crop_mode
        else:
            from .model import KilterViTGNN

            self.model = KilterViTGNN(checkpoint["config"]).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.crop_mode = "raw"

        self.embedder = HoldEmbedder(
            model_name=config["vit"]["model_name"],
            device=device_str,
        )

    def describe(self) -> dict:
        if self.version == 2:
            return {"version": 2, "members": len(self.ensemble.members), "crop_mode": self.crop_mode,
                    "strategy": self.ensemble.strategy}
        return {"version": 1, "crop_mode": "raw"}

    # ------------------------------------------------------------------
    def _crops(self, image_np, holds_info):
        if self.version == 2:
            from .holds_v2 import crop_hold

            return [crop_hold(image_np, hi["tap_x"], hi["tap_y"], mode=self.crop_mode)[0] for hi in holds_info]
        return [crop_hold_v1(image_np, hi["tap_x"], hi["tap_y"], self.config)[0] for hi in holds_info]

    def predict(
        self,
        image_np,
        holds_info: list[dict],
        wall_angle: float | None = None,
    ) -> dict:
        """Run full inference.

        Args:
            image_np: (H, W, 3) RGB uint8 array.
            holds_info: list of dicts with ``tap_x``, ``tap_y``, ``role``.
            wall_angle: degrees (default from config).

        Returns:
            dict with difficulty, v_grade, low, high, confidence_range, hold_crops.
        """
        if wall_angle is None:
            wall_angle = self.config["inference"]["default_wall_angle"]

        crops = self._crops(image_np, holds_info)
        embeddings = self.embedder.embed_batch(crops)

        if self.version == 2:
            out = self.ensemble.predict(
                embeddings,
                [(hi["tap_x"], hi["tap_y"]) for hi in holds_info],
                [hi["role"] for hi in holds_info],
                float(wall_angle),
                image_np.shape[:2],
            )
            prediction, lo, hi = out["difficulty"], out["low"], out["high"]
        else:
            from torch_geometric.loader import DataLoader

            from .graph import build_graph

            hold_data = [
                {"embedding": embeddings[i], "tap_x": hi["tap_x"], "tap_y": hi["tap_y"], "role": hi["role"]}
                for i, hi in enumerate(holds_info)
            ]
            data = build_graph(hold_data, wall_angle, image_np.shape)
            batch = next(iter(DataLoader([data], batch_size=1))).to(self.device)
            with torch.no_grad():
                prediction = self.model(batch).item()
            lo, hi = prediction - 1.0, prediction + 1.0

        v_grade = _difficulty_to_vgrade(prediction)
        low, high = _difficulty_to_vgrade(lo), _difficulty_to_vgrade(hi)
        return {
            "difficulty": float(prediction),
            "v_grade": v_grade,
            "low": low,
            "high": high,
            "confidence_range": f"{low} -- {high}",
            "hold_crops": crops,
        }
