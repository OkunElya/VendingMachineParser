"""
Image Classification Labeler
----------------------------
Opens a folder of images and lets you classify them into a YOLO classification dataset.

Usage:
    python image_labeler.py --images ./path/to/images --yaml ./dataset/data.yaml

Controls:
    n / Right  - Next image
    p / Left   - Previous image
    1-9        - Move image to class folder (train split)
    c          - Create a new class
    d          - Move image to _deleted folder
    q / ESC    - Quit

Updates:
    - nc in YAML is kept in sync with real class count
    - On start, images already in _deleted (sha256 match) are auto-moved
    - Non-square images are padded with black bars to fill a square
"""

import cv2
import yaml
import shutil
import hashlib
import argparse
import sys
import numpy as np
from pathlib import Path


# --- Config -------------------------------------------------------------------

MIN_W = 900
MIN_H = 700

BG_COLOR      = (30, 30, 30)
PANEL_COLOR   = (20, 20, 20)
ACCENT_COLOR  = (0, 200, 120)
TEXT_COLOR    = (220, 220, 220)
DIM_COLOR     = (100, 100, 100)
HOTKEY_COLOR  = (255, 200, 60)

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}


# --- YAML helpers -------------------------------------------------------------

def load_yaml(yaml_path: Path):
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def save_yaml(yaml_path: Path, data: dict):
    """Save YAML, always syncing nc to the real class count."""
    names = data.get('names', {})
    data['nc'] = len(names) if isinstance(names, list) else len(names)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def get_classes(data: dict) -> dict:
    """Return {0: 'name', 1: 'name', ...}"""
    names = data.get('names', {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def get_dataset_root(yaml_path: Path, data: dict) -> Path:
    path_field = data.get('path', None)
    if path_field:
        p = Path(path_field)
        if not p.is_absolute():
            p = yaml_path.parent / p
        return p.resolve()
    return yaml_path.parent.resolve()


# --- SHA-256 duplicate detection ---------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def build_deleted_hashes(images_folder: Path) -> set:
    """Return set of SHA-256 hashes for every file in _deleted/."""
    deleted_dir = images_folder / "_deleted"
    if not deleted_dir.exists():
        return set()
    hashes = set()
    for f in deleted_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT:
            hashes.add(sha256_file(f))
    return hashes


def filter_already_deleted(images: list, deleted_hashes: set,
                            images_folder: Path):
    """
    Move images whose hash matches a deleted file straight into _deleted/.
    Returns (remaining_images, skipped_count).
    """
    if not deleted_hashes:
        return images, 0

    kept = []
    skipped = 0
    deleted_dir = images_folder / "_deleted"
    deleted_dir.mkdir(exist_ok=True)

    for img_path in images:
        if sha256_file(img_path) in deleted_hashes:
            dest = deleted_dir / img_path.name
            shutil.move(str(img_path), str(dest))
            print(f"  -> Duplicate of deleted: {img_path.name}")
            skipped += 1
        else:
            kept.append(img_path)

    return kept, skipped


# --- Image helpers ------------------------------------------------------------

def collect_images(folder: Path) -> list:
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ])


def pad_to_square(img):
    """Pad image with black borders to make it square (centered)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    pad_top    = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left   = (size - w) // 2
    pad_right  = size - w - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def fit_image(img, min_w, min_h):
    """Scale image up if smaller than min_w x min_h, keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = max(min_w / w, min_h / h, 1.0)
    if scale > 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LINEAR)
    return img


# --- UI drawing ---------------------------------------------------------------

def draw_panel(canvas, classes: dict, current_idx: int, total: int,
               img_name: str, status_msg: str, input_mode: bool, input_text: str):
    panel_x = canvas.shape[1] - 260
    canvas[:, panel_x:] = PANEL_COLOR

    def put(text, y, color=TEXT_COLOR, scale=0.5, thickness=1):
        cv2.putText(canvas, text, (panel_x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    # Title bar
    cv2.rectangle(canvas, (panel_x, 0), (canvas.shape[1], 48), ACCENT_COLOR, -1)
    cv2.putText(canvas, "IMAGE LABELER", (panel_x + 14, 31),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 10), 1, cv2.LINE_AA)

    # Counter + filename
    put(f"{current_idx + 1} / {total}", 72, HOTKEY_COLOR, 0.55, 1)
    name = img_name if len(img_name) <= 28 else img_name[:25] + "..."
    put(name, 92, DIM_COLOR, 0.42, 1)

    cv2.line(canvas, (panel_x + 10, 108), (canvas.shape[1] - 10, 108), (60, 60, 60), 1)

    # Class list
    put("CLASSES", 130, ACCENT_COLOR, 0.45, 1)
    y = 152
    for class_id, class_name in sorted(classes.items()):
        key_char = str(class_id + 1) if class_id < 9 else "-"
        cv2.putText(canvas, f"[{key_char}]", (panel_x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, HOTKEY_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, class_name, (panel_x + 48, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)
        y += 22
        if y > canvas.shape[0] - 130:
            put("... (more)", y, DIM_COLOR, 0.42)
            break

    cv2.line(canvas, (panel_x + 10, canvas.shape[0] - 115),
             (canvas.shape[1] - 10, canvas.shape[0] - 115), (60, 60, 60), 1)

    # Controls
    for key_str, desc in [("[n/p]", "Next / Prev"), ("[c]", "New class"),
                           ("[d]",   "Delete"),      ("[q]", "Quit")]:
        cy = canvas.shape[0] - 100 + [("[n/p]", "Next / Prev"), ("[c]", "New class"),
                                       ("[d]", "Delete"), ("[q]", "Quit")].index(
                                           (key_str, desc)) * 18
        cv2.putText(canvas, key_str, (panel_x + 12, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, HOTKEY_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, desc, (panel_x + 52, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM_COLOR, 1, cv2.LINE_AA)

    # Status
    if status_msg:
        sw = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        sx = max(panel_x + 12, panel_x + (248 - sw) // 2)
        cv2.putText(canvas, status_msg, (sx, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_COLOR, 1, cv2.LINE_AA)

    # Input overlay
    if input_mode:
        ox, oy, ow, oh = panel_x + 10, canvas.shape[0] // 2 - 40, 240, 80
        cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), (50, 50, 50), -1)
        cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), ACCENT_COLOR, 1)
        cv2.putText(canvas, "New class name:", (ox + 8, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, ACCENT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, input_text + "|", (ox + 8, oy + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, HOTKEY_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Enter=confirm  Esc=cancel", (ox + 8, oy + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM_COLOR, 1, cv2.LINE_AA)


def build_canvas(img, classes, current_idx, total, img_name,
                 status_msg, input_mode, input_text):
    panel_w  = 260
    # 1. Pad non-square images to square with black bars
    img_sq   = pad_to_square(img)
    # 2. Scale up if too small
    img_fit  = fit_image(img_sq, MIN_W, MIN_H)
    ih, iw   = img_fit.shape[:2]
    canvas_h = max(ih, MIN_H)

    canvas = img_fit.copy() if ih == canvas_h else \
        cv2.copyMakeBorder(img_fit, 0, canvas_h - ih, 0, 0,
                           cv2.BORDER_CONSTANT, value=BG_COLOR)

    panel_blank = np.zeros((canvas_h, panel_w, 3), dtype='uint8')
    canvas = np.hstack([canvas, panel_blank])

    draw_panel(canvas, classes, current_idx, total, img_name,
               status_msg, input_mode, input_text)
    return canvas


# --- File operations ----------------------------------------------------------

def move_to_class(img_path: Path, class_name: str, dataset_root: Path):
    dest_dir = dataset_root / "train" / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / img_path.name
    if dest.exists():
        dest = dest_dir / (img_path.stem + "_dup" + img_path.suffix)
    shutil.move(str(img_path), str(dest))


def move_to_deleted(img_path: Path):
    deleted_dir = img_path.parent / "_deleted"
    deleted_dir.mkdir(exist_ok=True)
    shutil.move(str(img_path), str(deleted_dir / img_path.name))


# --- Main loop ----------------------------------------------------------------

def run(images_folder: Path, yaml_path: Path):
    if not yaml_path.exists():
        print(f"ERROR: YAML not found: {yaml_path}")
        sys.exit(1)

    yaml_data = load_yaml(yaml_path)
    classes   = get_classes(yaml_data)
    ds_root   = get_dataset_root(yaml_path, yaml_data)

    print(f"Images folder : {images_folder}")
    print(f"Dataset YAML  : {yaml_path}")
    print(f"Dataset root  : {ds_root}")
    print(f"Classes       : {classes}")

    # Sync nc on startup
    save_yaml(yaml_path, yaml_data)
    print(f"nc synced to {yaml_data['nc']}")

    images = collect_images(images_folder)
    if not images:
        print("ERROR: No images found in the folder.")
        sys.exit(1)

    # Check for duplicates of already-deleted images
    print(f"Checking {len(images)} images against _deleted hashes...")
    deleted_hashes = build_deleted_hashes(images_folder)
    if deleted_hashes:
        images, skipped = filter_already_deleted(images, deleted_hashes, images_folder)
        if skipped:
            print(f"  Auto-moved {skipped} duplicate(s) of deleted images.")
    else:
        print("  No _deleted folder yet — skipping hash check.")

    if not images:
        print("All images matched deleted ones. Nothing left to label.")
        sys.exit(0)

    print(f"{len(images)} images ready to label.")

    idx         = 0
    status_msg  = ""
    input_mode  = False
    input_text  = ""
    window_name = "Image Labeler"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        images = collect_images(images_folder)
        if not images:
            print("No more images to label.")
            break

        idx = max(0, min(idx, len(images) - 1))
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            status_msg = "Could not load image"
            idx += 1
            continue

        canvas = build_canvas(img, classes, idx, len(images),
                              img_path.name, status_msg, input_mode, input_text)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(50) & 0xFF
        if key == 255:
            continue

        # Input mode (new class)
        if input_mode:
            if key == 13:  # Enter
                new_name = input_text.strip()
                if new_name:
                    new_id = max(classes.keys(), default=-1) + 1
                    classes[new_id] = new_name
                    yaml_data['names'] = classes
                    save_yaml(yaml_path, yaml_data)
                    status_msg = f"Added [{new_id + 1}] {new_name}  nc={yaml_data['nc']}"
                else:
                    status_msg = "Empty name, cancelled."
                input_mode = False
                input_text = ""
            elif key == 27:  # Esc
                input_mode = False
                input_text = ""
                status_msg = "Cancelled."
            elif key == 8:  # Backspace
                input_text = input_text[:-1]
            elif 32 <= key <= 126:
                input_text += chr(key)
            continue

        # Navigation
        if key in (ord('n'), 83):
            idx = (idx + 1) % len(images)
            status_msg = ""
        elif key in (ord('p'), 81):
            idx = (idx - 1) % len(images)
            status_msg = ""

        # Delete
        elif key == ord('d'):
            move_to_deleted(img_path)
            status_msg = f"Deleted: {img_path.name}"

        # New class
        elif key == ord('c'):
            input_mode = True
            input_text = ""
            status_msg = ""

        # Quit
        elif key in (ord('q'), 27):
            break

        # Class assignment 1-9
        elif ord('1') <= key <= ord('9'):
            class_idx = key - ord('1')
            if class_idx in classes:
                move_to_class(img_path, classes[class_idx], ds_root)
                status_msg = f"-> {classes[class_idx]}"
            else:
                status_msg = f"No class [{class_idx + 1}]"

    cv2.destroyAllWindows()
    print("Labeling session ended.")


# --- Entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive image labeler for YOLO classification datasets."
    )
    parser.add_argument("--images", "-i", type=str, required=True,
                        help="Folder containing images to label")
    parser.add_argument("--yaml", "-y", type=str, required=True,
                        help="Path to classification dataset data.yaml")
    args = parser.parse_args()

    run(
        images_folder=Path(args.images).expanduser().resolve(),
        yaml_path=Path(args.yaml).expanduser().resolve()
    )


if __name__ == "__main__":
    main()