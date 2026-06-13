"""
Migrate ./gallery/<class>/*.png from the old square, edge-mean-padded
crop_obb_rotated(pad_square=True) convention to the new unpadded
crop_obb_rotated(pad_square=False) convention.

crop_obb_rotated canonicalises crops to landscape (w >= h) *before* padding,
so any square-padding it added is always one or two horizontal bands at the
top and/or bottom, each band a single constant color (the crop's
edge-mean color), with |pad_b - pad_t| <= 1. This script detects and strips
those bands, leaving the original (h_i, w_i) rectangular crop -- which
ProductBank._embed now black-pads itself (matching training).

Images where no such band is found are left untouched. Images whose top/
bottom bands don't fit the expected pattern (inconsistent sizes/colors, or
an implausibly large band) are left untouched and reported for manual
review, since stripping them could cut into real product content.

The whole gallery directory is backed up (once) before any file is
overwritten.

Usage:
    python scripts/migrate_gallery_padding.py [--gallery ./gallery] [--backup ./gallery_backup]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "python"))

from shared import GALLERY_DIR, IMAGE_EXTS


def _count_uniform_rows(img: np.ndarray, from_top: bool) -> tuple[int, np.ndarray | None]:
    """Count consecutive rows from the top (or bottom) that are each a
    single constant color equal to the first such row's color. Returns
    (count, color) -- color is None if count == 0."""
    h = img.shape[0]
    rows = range(h) if from_top else range(h - 1, -1, -1)
    color = None
    count = 0
    for r in rows:
        row = img[r]
        if color is None:
            if not np.all(row == row[0]):
                break
            color = row[0].copy()
        elif not np.all(row == color):
            break
        count += 1
    return count, color


def detect_padding(img: np.ndarray) -> tuple[int, int] | None:
    """Return (pad_t, pad_b) rows to strip, or None if the image should be
    left untouched (no padding, or the pattern doesn't look like genuine
    crop_obb_rotated padding)."""
    h = img.shape[0]
    pad_t, color_t = _count_uniform_rows(img, from_top=True)
    pad_b, color_b = _count_uniform_rows(img, from_top=False)

    if pad_t == 0 and pad_b == 0:
        return None

    diff = pad_b - pad_t
    if not (0 <= diff <= 1):
        return None
    if pad_t > 0 and pad_b > 0 and not np.array_equal(color_t, color_b):
        return None
    if h - pad_t - pad_b < 1:
        return None

    return pad_t, pad_b


def migrate(gallery_dir: Path, backup_dir: Path) -> None:
    if not backup_dir.exists():
        print(f"Backing up {gallery_dir} -> {backup_dir}")
        shutil.copytree(gallery_dir, backup_dir)
    else:
        print(f"Backup {backup_dir} already exists, skipping backup")

    stripped = unchanged = flagged = 0
    flagged_paths: list[Path] = []

    for class_dir in sorted(gallery_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_stripped = class_unchanged = class_flagged = 0

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if img.shape[0] != img.shape[1]:
                class_unchanged += 1
                continue

            padding = detect_padding(img)
            if padding is None:
                pad_t, _ = _count_uniform_rows(img, from_top=True)
                pad_b, _ = _count_uniform_rows(img, from_top=False)
                if pad_t == 0 and pad_b == 0:
                    class_unchanged += 1
                else:
                    class_flagged += 1
                    flagged_paths.append(img_path)
                    print(f"  FLAGGED {img_path}: pad_t={pad_t} pad_b={pad_b} side={img.shape[0]}")
                continue

            pad_t, pad_b = padding
            if pad_t == 0 and pad_b == 0:
                class_unchanged += 1
                continue

            h = img.shape[0]
            cropped = img[pad_t:h - pad_b, :, :]
            cv2.imwrite(str(img_path), cropped)
            class_stripped += 1

        print(f"{class_dir.name}: stripped={class_stripped} unchanged={class_unchanged} flagged={class_flagged}")
        stripped += class_stripped
        unchanged += class_unchanged
        flagged += class_flagged

    print(f"\nTotal: stripped={stripped} unchanged={unchanged} flagged={flagged}")
    if flagged_paths:
        print("\nReview these manually (left untouched):")
        for p in flagged_paths:
            print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", default=GALLERY_DIR, help=f"Gallery directory (default: {GALLERY_DIR})")
    parser.add_argument("--backup", default=None, help="Backup directory (default: <gallery>_backup)")
    args = parser.parse_args()

    gallery_dir = Path(args.gallery).expanduser().resolve()
    backup_dir  = Path(args.backup).expanduser().resolve() if args.backup \
        else gallery_dir.with_name(gallery_dir.name + "_backup")

    if not gallery_dir.is_dir():
        sys.exit(f"Not a directory: {gallery_dir}")

    migrate(gallery_dir, backup_dir)


if __name__ == "__main__":
    main()
