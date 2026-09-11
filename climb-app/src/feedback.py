"""Feedback persistence -- saves user grade corrections with images.

Each submission is stored as:
    feedback/submissions/{id}/metadata.json
    feedback/submissions/{id}/wall_photo.jpg

On HF Spaces the entire submissions/ folder is synced to a HF Dataset
repo when HF_TOKEN and FEEDBACK_HF_REPO are set.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

_FEEDBACK_ROOT = Path("feedback")
_SUBMISSIONS_DIR = _FEEDBACK_ROOT / "submissions"
_IMAGE_FILENAME = "wall_photo.jpg"


def save_feedback(
    entry: dict,
    image: Image.Image | None = None,
    sync: bool = True,
) -> tuple[str, str]:
    """Save a feedback entry with an optional wall photo.

    *entry* contains holds, wall_angle, predicted/actual grades, etc.
    *image* is the original uploaded PIL image (before any annotation).
    When *sync* is False the HuggingFace upload is skipped; call
    ``sync_submission`` later (e.g. from a background thread).

    Returns (submission_id, status_message).
    """
    submission_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    sub_dir = _SUBMISSIONS_DIR / submission_id
    sub_dir.mkdir(parents=True, exist_ok=True)

    image_w: int | None = None
    image_h: int | None = None

    if image is not None:
        image_w, image_h = image.size
        image.save(sub_dir / _IMAGE_FILENAME, format="JPEG", quality=90)

    metadata = {
        "id": submission_id,
        "timestamp": timestamp,
        **({
            "image_filename": _IMAGE_FILENAME,
            "image_width": image_w,
            "image_height": image_h,
        } if image is not None else {}),
        **entry,
    }

    with open(sub_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not sync:
        return submission_id, "Feedback saved locally."
    status = _sync_to_hf(submission_id, sub_dir, image is not None)
    return submission_id, status


def sync_submission(submission_id: str) -> str:
    """Upload (or re-upload) everything in a submission folder to HuggingFace."""
    sub_dir = _SUBMISSIONS_DIR / submission_id
    if not (sub_dir / "metadata.json").exists():
        return "Submission not found."
    return _sync_to_hf(submission_id, sub_dir, (sub_dir / _IMAGE_FILENAME).exists())


def _sync_to_hf(submission_id: str, sub_dir: Path, has_image: bool) -> str:
    """Upload the submission folder to HuggingFace."""
    hf_repo = os.environ.get("FEEDBACK_HF_REPO")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_repo or not hf_token:
        return "Feedback saved locally."

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        files_to_upload = [
            (str(sub_dir / "metadata.json"),
             f"submissions/{submission_id}/metadata.json"),
        ]
        if has_image:
            files_to_upload.append(
                (str(sub_dir / _IMAGE_FILENAME),
                 f"submissions/{submission_id}/{_IMAGE_FILENAME}"),
            )

        for local_path, repo_path in files_to_upload:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=hf_repo,
                repo_type="dataset",
            )
        return "Feedback saved and synced."
    except Exception:
        return "Feedback saved locally (sync failed)."


def update_feedback(submission_id: str, actual_grade: str, sync: bool = True) -> str:
    """Patch an existing submission with the user-provided actual grade.

    Re-uploads only the updated metadata.json to HuggingFace.
    """
    sub_dir = _SUBMISSIONS_DIR / submission_id
    meta_path = sub_dir / "metadata.json"
    if not meta_path.exists():
        return "Submission not found."

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["actual_grade"] = actual_grade
    metadata["feedback_timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not sync:
        return "Grade updated locally."

    hf_repo = os.environ.get("FEEDBACK_HF_REPO")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_repo or not hf_token:
        return "Grade updated locally."

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo=f"submissions/{submission_id}/metadata.json",
            repo_id=hf_repo,
            repo_type="dataset",
        )
        return "Grade updated and synced."
    except Exception:
        return "Grade updated locally (sync failed)."


def load_feedback() -> list[dict]:
    """Load all feedback entries from local submission folders."""
    if not _SUBMISSIONS_DIR.exists():
        return []
    entries = []
    for meta_file in sorted(_SUBMISSIONS_DIR.glob("*/metadata.json")):
        try:
            entries.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return entries
