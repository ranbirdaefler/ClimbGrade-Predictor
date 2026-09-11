"""Re-run inference on gym dataset with fixed cropping.

Downloads submissions from HuggingFace, runs the full prediction
pipeline with the new resolution-adaptive crops, and compares
old vs new predictions against actual grades.

Usage: python scripts/repredict_gym.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.predict import GradePredictor


def main() -> None:
    dataset_path = snapshot_download(
        repo_id="ranbirr1/climb-route-w-grade",
        repo_type="dataset",
        local_dir="data/gym_dataset",
    )

    submissions_dir = Path(dataset_path) / "submissions"

    with open("configs/inference.yaml") as f:
        config = yaml.safe_load(f)

    print("Loading model...")
    predictor = GradePredictor(config)

    results: list[dict] = []

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

        holds_info = [
            {"tap_x": h["tap_x"], "tap_y": h["tap_y"], "role": h["role"]}
            for h in meta["holds"]
        ]

        result = predictor.predict(image_np, holds_info, meta["wall_angle"])

        actual = meta.get("actual_grade", "?")
        old_pred = meta.get("predicted_grade", "?")
        new_pred = result["v_grade"]

        print(f"\nRoute {meta['id'][:8]}:")
        print(f"  Actual:         {actual}")
        print(f"  Old prediction: {old_pred}")
        print(f"  New prediction: {new_pred}")
        print(f"  Raw difficulty: {result['difficulty']:.1f}")
        print(f"  Confidence:     {result['confidence_range']}")

        results.append({
            "id": meta["id"],
            "actual": actual,
            "old_predicted": old_pred,
            "new_predicted": new_pred,
            "new_difficulty": round(result["difficulty"], 2),
        })

    print("\n" + "=" * 60)
    print("REPREDICTION SUMMARY")
    print("=" * 60)
    print(f"{'ID':>12}  {'Actual':>8}  {'Old':>8}  {'New':>8}  {'Diff':>6}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['id'][:12]:>12}  {r['actual']:>8}  "
            f"{r['old_predicted']:>8}  {r['new_predicted']:>8}  "
            f"{r['new_difficulty']:>6.1f}"
        )

    output_path = Path("results/repredict_gym.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
