"""FastAPI backend for V2 In My Gym -- the AI climbing grade predictor.

Serves the static single-page frontend from ./static and exposes a small
JSON API around the DINOv2 + GATv2 inference pipeline in src/inference.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import base64
import io
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from src.feedback import save_feedback, sync_submission, update_feedback
from src.inference.crop import crop_hold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2inmygym")

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_SIDE = 1800          # downscale huge phone photos before inference
SESSION_TTL_S = 60 * 60        # forget an uploaded photo after an hour idle
MAX_SESSIONS = 200
MIN_HOLDS, MAX_HOLDS = 3, 40
VALID_ROLES = {"start", "hand", "finish", "foot", "volume"}
GRADE_RE = re.compile(r"^V(?:[0-9]|1[0-3])$")
SUBMISSION_RE = re.compile(r"^[0-9a-f]{12}$")


def _load_config() -> dict:
    with open(ROOT / "configs" / "inference.yaml") as f:
        return yaml.safe_load(f)


_config = _load_config()

# ── Model state ──────────────────────────────────────────────────────
_model: dict = {"predictor": None, "error": None, "device": None, "load_s": None}
_infer_lock = threading.Lock()


def _warm_up() -> None:
    t0 = time.perf_counter()
    try:
        from src.inference.predict import GradePredictor

        predictor = GradePredictor(_config)
        _model["device"] = str(predictor.device)
        _model["predictor"] = predictor
        _model["load_s"] = round(time.perf_counter() - t0, 1)
        log.info("model ready on %s in %.1fs", predictor.device, _model["load_s"])
    except Exception as exc:  # noqa: BLE001 - surfaced through /api/health
        log.exception("model failed to load")
        _model["error"] = f"{type(exc).__name__}: {exc}"


# ── Photo sessions (in-memory, single replica) ───────────────────────
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _put_session(image_pil: Image.Image) -> str:
    sid = uuid.uuid4().hex[:16]
    now = time.time()
    with _sessions_lock:
        expired = [k for k, v in _sessions.items() if now - v["last"] > SESSION_TTL_S]
        for k in expired:
            del _sessions[k]
        while len(_sessions) >= MAX_SESSIONS:
            oldest = min(_sessions, key=lambda k: _sessions[k]["last"])
            del _sessions[oldest]
        _sessions[sid] = {
            "pil": image_pil,
            "np": np.asarray(image_pil),
            "last": now,
            "fingerprint": None,
            "submission_id": None,
        }
    return sid


def _get_session(sid: str) -> dict:
    with _sessions_lock:
        session = _sessions.get(sid)
        if session is None:
            raise HTTPException(404, "That photo session expired. Upload the photo again.")
        session["last"] = time.time()
        return session


# ── Helpers ──────────────────────────────────────────────────────────
# Outlines bigger than this fraction of the photo are wall, not hold: the
# segmentation still produces a usable crop, but drawing them looks wrong.
MAX_OUTLINE_SIDE_FRAC = 0.30
MAX_OUTLINE_AREA_FRAC = 0.05


def _mask_contour(mask: np.ndarray | None) -> list[list[int]] | None:
    """Largest external contour of the segmentation mask, simplified.

    Returns None when there is no mask or when the region is implausibly
    large for a single hold (the colour flood escaped onto the wall).
    """
    if mask is None:
        return None
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    if (xs.max() - xs.min()) > MAX_OUTLINE_SIDE_FRAC * w:
        return None
    if (ys.max() - ys.min()) > MAX_OUTLINE_SIDE_FRAC * h:
        return None
    if len(xs) > MAX_OUTLINE_AREA_FRAC * w * h:
        return None
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.008 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    if len(contour) < 3:
        return None
    return [[int(p[0][0]), int(p[0][1])] for p in contour]


def _thumb_data_url(crop: Image.Image, size: int = 112) -> str:
    thumb = crop.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── Schemas ──────────────────────────────────────────────────────────
class HoldLocateIn(BaseModel):
    session_id: str
    x: int
    y: int


class HoldIn(BaseModel):
    x: int
    y: int
    role: str = "hand"


class PredictIn(BaseModel):
    session_id: str
    holds: list[HoldIn] = Field(min_length=MIN_HOLDS, max_length=MAX_HOLDS)
    wall_angle: float = 30


class FeedbackIn(BaseModel):
    submission_id: str
    actual_grade: str


def _clean_hold(hold: HoldIn, width: int, height: int) -> dict:
    return {
        "tap_x": int(min(max(hold.x, 0), width - 1)),
        "tap_y": int(min(max(hold.y, 0), height - 1)),
        "role": hold.role if hold.role in VALID_ROLES else "hand",
    }


# ── App ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm_up, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(title="V2 In My Gym", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    if _model["error"]:
        return {"status": "error", "detail": _model["error"]}
    if _model["predictor"] is None:
        return {"status": "loading"}
    return {"status": "ready", "device": _model["device"], "load_s": _model["load_s"],
            "model": _model["predictor"].describe()}


@app.post("/api/session")
async def create_session(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo is too large (25 MB max).")
    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "That file doesn't look like an image.") from None

    w, h = image.size
    scale = min(1.0, MAX_IMAGE_SIDE / max(w, h))
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    sid = _put_session(image)
    return {"session_id": sid, "width": image.width, "height": image.height}


@app.post("/api/hold")
def locate_hold(req: HoldLocateIn) -> dict:
    session = _get_session(req.session_id)
    image = session["np"]
    h, w = image.shape[:2]
    if not (0 <= req.x < w and 0 <= req.y < h):
        raise HTTPException(400, "Tap is outside the photo.")
    crop, mask, bbox = crop_hold(image, req.x, req.y, _config)
    contour = _mask_contour(mask)
    return {
        "detected": contour is not None,
        "bbox": [int(v) for v in bbox],
        "contour": contour,
        "thumb": _thumb_data_url(crop),
    }


@app.post("/api/predict")
def predict(req: PredictIn) -> dict:
    if _model["error"]:
        raise HTTPException(503, "The model failed to load on this server.")
    predictor = _model["predictor"]
    if predictor is None:
        raise HTTPException(503, "The model is still warming up. Try again in a moment.")

    session = _get_session(req.session_id)
    image = session["np"]
    h, w = image.shape[:2]
    holds_info = [_clean_hold(hd, w, h) for hd in req.holds]
    wall_angle = float(min(max(req.wall_angle, 0.0), 70.0))

    t0 = time.perf_counter()
    with _infer_lock:
        result = predictor.predict(image, holds_info, wall_angle)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    difficulty = float(result["difficulty"])
    low, high = result["low"], result["high"]

    # Persist once per unique (holds, angle) so repeated clicks don't duplicate.
    fingerprint = repr([(hd["tap_x"], hd["tap_y"], hd["role"]) for hd in holds_info]) + repr(wall_angle)
    if fingerprint != session["fingerprint"]:
        entry = {
            "holds": holds_info,
            "wall_angle": wall_angle,
            "predicted_grade": result["v_grade"],
            "predicted_difficulty": round(difficulty, 3),
        }
        sub_id, _ = save_feedback(entry, image=session["pil"], sync=False)
        threading.Thread(target=sync_submission, args=(sub_id,), daemon=True).start()
        session["submission_id"] = sub_id
        session["fingerprint"] = fingerprint

    return {
        "v_grade": result["v_grade"],
        "difficulty": round(difficulty, 3),
        "low": low,
        "high": high,
        "n_holds": len(holds_info),
        "wall_angle": wall_angle,
        "elapsed_ms": elapsed_ms,
        "submission_id": session["submission_id"],
    }


@app.post("/api/feedback")
def feedback(req: FeedbackIn) -> dict:
    if not GRADE_RE.match(req.actual_grade):
        raise HTTPException(400, "Grade must be V0 to V13.")
    if not SUBMISSION_RE.match(req.submission_id):
        raise HTTPException(400, "Bad submission id.")
    message = update_feedback(req.submission_id, req.actual_grade, sync=False)
    if message == "Submission not found.":
        raise HTTPException(404, message)
    threading.Thread(target=sync_submission, args=(req.submission_id,), daemon=True).start()
    return {"ok": True, "message": message}


# ── Static frontend ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})
