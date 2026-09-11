"""Full inference pipeline: image + tap points -> grade prediction."""

from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from .crop import crop_hold
from .embed import HoldEmbedder
from .graph import build_graph
from .model import KilterViTGNN

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

        checkpoint = torch.load(
            config["model"]["checkpoint"],
            map_location=self.device,
            weights_only=False,
        )
        model_config = checkpoint["config"]
        self.model = KilterViTGNN(model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.embedder = HoldEmbedder(
            model_name=config["vit"]["model_name"],
            device=device_str,
        )

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
            dict with difficulty, v_grade, confidence_range, hold_crops, hold_masks.
        """
        if wall_angle is None:
            wall_angle = self.config["inference"]["default_wall_angle"]

        crops, bboxes = [], []
        for hi in holds_info:
            crop_pil, _mask, bbox = crop_hold(
                image_np, hi["tap_x"], hi["tap_y"], self.config
            )
            crops.append(crop_pil)
            bboxes.append(bbox)

        embeddings = self.embedder.embed_batch(crops)

        hold_data = []
        for i, hi in enumerate(holds_info):
            hold_data.append({
                "embedding": embeddings[i],
                "tap_x": hi["tap_x"],
                "tap_y": hi["tap_y"],
                "role": hi["role"],
            })

        data = build_graph(hold_data, wall_angle, image_np.shape)
        loader = DataLoader([data], batch_size=1)
        batch = next(iter(loader)).to(self.device)

        with torch.no_grad():
            prediction = self.model(batch).item()

        v_grade = _difficulty_to_vgrade(prediction)
        low = _difficulty_to_vgrade(prediction - 1.0)
        high = _difficulty_to_vgrade(prediction + 1.0)

        return {
            "difficulty": prediction,
            "v_grade": v_grade,
            "confidence_range": f"{low} -- {high}",
            "hold_crops": crops,
        }
