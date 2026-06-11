from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from grid_helper import build_markdown_table, render_grid
from pipeline import MachineDetection, Pipeline

_TOKENS_PATH  = "./tokens.yaml"
_GALLERY_DIR  = Path("./gallery")
_WEB_DIST_DIR = Path(".src/web/dist")
_UPLOADS_DIR  = Path("./uploads")
_IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _load_tokens() -> set[str]:
    with open(_TOKENS_PATH) as f:
        data = yaml.safe_load(f) or {}
    return set((data.get("tokens") or {}).values())


_valid_tokens = _load_tokens()
_bearer       = HTTPBearer()


def require_token(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    if creds.credentials not in _valid_tokens:
        raise HTTPException(status_code=401, detail="Invalid access token")
    return creds.credentials


app      = FastAPI(title="Vending Machine Parser API")
pipeline = Pipeline()


def _serialize_detection(det: MachineDetection) -> dict:
    grid_data = None
    if det.grid is not None:
        overlay = render_grid(det.image, det.window_points, det.grid)
        ok, buf = cv2.imencode(".jpg", overlay)
        grid_data = {
            "n_rows":   det.grid.n_rows,
            "n_cols":   det.grid.n_cols,
            "markdown": build_markdown_table(det.grid),
            "cells": [
                {
                    "row":           cell.row,
                    "col":           cell.col,
                    "col_span":      cell.col_span,
                    "product_name":  cell.product_name,
                    "product_score": cell.product_score,
                }
                for cell in det.grid.cells
            ],
            "image": base64.b64encode(buf).decode("ascii") if ok else None,
        }

    return {
        "machine_info": det.machine_info,
        "bbox":         [round(float(v), 1) for v in det.machine_bbox.tolist()],
        "grid":         grid_data,
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...), _token: str = Depends(require_token)):
    """Run the detection pipeline on an uploaded image and return the parsed grid(s)."""
    raw   = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

    status = "OK" if image is not None else "FAIL"
    suffix = Path(file.filename or "").suffix or ".jpg"
    name   = f"{status}_{_token}_{int(time.time() * 1000)}{suffix}"
    (_UPLOADS_DIR / name).write_bytes(raw)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detections = pipeline.detect(image)
    return {"machines": [_serialize_detection(det) for det in detections]}


@app.get("/products/{name}/image")
def product_image(name: str, _token: str = Depends(require_token)):
    """Return a reference image for the product gallery class matching `name`."""
    match = next(
        (d for d in _GALLERY_DIR.iterdir() if d.is_dir() and d.name.lower() == name.lower()),
        None,
    )
    if match is None:
        raise HTTPException(status_code=404, detail=f"Unknown product '{name}'")

    image_path = next(
        (p for p in sorted(match.iterdir()) if p.suffix.lower() in _IMAGE_EXTS),
        None,
    )
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"No image found for product '{name}'")

    return FileResponse(image_path)


# Serve the built frontend (run `npm run build` in web/) from the same
# uvicorn process — same host and port as the API, so no CORS is needed.
if _WEB_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST_DIR, html=True), name="web")

