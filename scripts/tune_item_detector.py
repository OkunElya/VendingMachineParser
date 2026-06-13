"""
Item Detector Tuner
--------------------
Interactive slider tool for tuning the item detector's confidence and NMS-IoU
thresholds, plus the post-detection merge thresholds (`ITEM_MERGE_IOU`,
`ITEM_MERGE_CONTAINMENT`).

For each image, runs `Pipeline.detect(image, classify=False)` once to get the
machine crop and rectified window (machine detect -> classify -> window
segment), masks the window the same way `Pipeline._detect_items_in_window`
does, then on every slider change re-runs only the item detector + merge step
and draws the resulting OBBs.

Usage:
    python scripts/tune_item_detector.py
    python scripts/tune_item_detector.py --input path/to/image.jpg
    python scripts/tune_item_detector.py --input path/to/image_dir

Controls:
    Conf / NMS IoU / Merge IoU / Merge Containment sliders -- live preview
    n / p     Next / previous image (when --input is a directory)
    s         Save the current slider values to config.json (via shared.CONFIG)
    q / Esc   Quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "python"))

from grid_helper import draw_obbs, mask_to_polygon, merge_overlapping_items, offset_obb
from pipeline import Pipeline
from shared import CONFIG, IMAGE_EXTS, ITEM_DETECTOR_IMGSZ

DEFAULT_IMAGE = "datasets/vending_machine_detection/train/images/PXL_20260610_043117947.jpg"
WINDOW_NAME   = "Item Detector Tuner"
MAX_DISPLAY   = 1000
BAR_H         = 56

SLIDERS = [
    ("Conf x100",       "item_detector_conf"),
    ("NMS IoU x100",    "item_detector_iou"),
    ("Merge IoU x100",  "item_merge_iou"),
    ("Merge Cont x100", "item_merge_containment"),
]


def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def detect_obbs(item_detector, image, conf: float, iou: float) -> np.ndarray:
    """Run the item detector with explicit (slider-controlled) thresholds.
    Mirrors Pipeline._detect_items, but conf/iou aren't fixed module constants."""
    result = item_detector.predict(image, verbose=False, conf=conf, iou=iou, imgsz=ITEM_DETECTOR_IMGSZ)[0]
    if result.obb is not None:
        return result.obb.xywhr.cpu().numpy().astype(np.float32)
    boxes = result.boxes.xywh.cpu().numpy().astype(np.float32)
    return np.concatenate([boxes, np.zeros((boxes.shape[0], 1), dtype=np.float32)], axis=1)


def fit_for_display(image: np.ndarray, max_dim: int = MAX_DISPLAY) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def load_image_state(pipeline: Pipeline, path: Path):
    """Run the pipeline up through window segmentation for one image.
    Returns (det, masked, (ox, oy)), or None if no machine/window was found."""
    image = cv2.imread(str(path))
    if image is None:
        print(f"Could not read image: {path}")
        return None
    results = pipeline.detect(image, classify=False)
    if not results:
        print(f"No machine detected in: {path.name}")
        return None
    det = results[0]
    masked, offset = mask_to_polygon(det.image, det.window_points)
    return det, masked, offset


def main() -> None:
    parser = argparse.ArgumentParser(description="Slider tuner for item detector + merge thresholds.")
    parser.add_argument("--input", "-i", default=DEFAULT_IMAGE,
                         help=f"Image file, or directory of images to step through with n/p (default: {DEFAULT_IMAGE})")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if input_path.is_dir():
        images = collect_images(input_path)
        if not images:
            sys.exit(f"No images found in {input_path}")
    elif input_path.is_file():
        images = [input_path]
    else:
        sys.exit(f"Not a file or directory: {input_path}")

    print("Loading models...")
    pipeline = Pipeline()

    img_idx = 0
    state = load_image_state(pipeline, images[img_idx])
    while state is None and img_idx + 1 < len(images):
        img_idx += 1
        state = load_image_state(pipeline, images[img_idx])
    if state is None:
        sys.exit("No machine detected in any input image.")
    det, masked, (ox, oy) = state

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    for label, field in SLIDERS:
        cv2.createTrackbar(label, WINDOW_NAME, round(getattr(CONFIG, field) * 100), 100, lambda _v: None)

    last_values = None
    while True:
        key = cv2.waitKey(50) & 0xFF

        if key in (ord("n"), ord("p")) and len(images) > 1:
            step     = 1 if key == ord("n") else -1
            new_idx  = (img_idx + step) % len(images)
            new_state = load_image_state(pipeline, images[new_idx])
            if new_state is not None:
                img_idx = new_idx
                det, masked, (ox, oy) = new_state
                last_values = None

        values = tuple(cv2.getTrackbarPos(label, WINDOW_NAME) / 100.0 for label, _ in SLIDERS)
        if values != last_values:
            conf, nms_iou, merge_iou, merge_containment = values

            obbs  = detect_obbs(pipeline.item_detector, masked, conf, nms_iou)
            items = [{"obb": offset_obb(o, ox, oy)} for o in obbs]
            merged = merge_overlapping_items(items, merge_iou, merge_containment)

            vis = det.image.copy()
            draw_obbs(vis, [item["obb"] for item in merged], color=(0, 255, 0), thickness=2)
            vis = fit_for_display(vis)

            cv2.rectangle(vis, (0, 0), (vis.shape[1], BAR_H), (0, 0, 0), -1)
            cv2.putText(vis, f"[{img_idx + 1}/{len(images)}] {images[img_idx].name}  raw={len(obbs)} merged={len(merged)}",
                         (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"conf={conf:.2f}  nms_iou={nms_iou:.2f}  merge_iou={merge_iou:.2f}  merge_cont={merge_containment:.2f}"
                              f"   (n/p: image, s: save, q: quit)",
                         (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)
            last_values = values

        if key == ord("s"):
            conf, nms_iou, merge_iou, merge_containment = values
            CONFIG.item_detector_conf     = conf
            CONFIG.item_detector_iou      = nms_iou
            CONFIG.item_merge_iou         = merge_iou
            CONFIG.item_merge_containment = merge_containment
            CONFIG.save()
            print(f"Saved to {Path('config.json').resolve()}: {CONFIG}")
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
