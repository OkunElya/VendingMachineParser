"""
Verify every image in a YOLO-style dataset with OpenCV, re-save it (re-encoding
fixes corrupt/truncated JPEG byte data), and drop any image (+ its label) that
fails to load or doesn't have sane dimensions.

Usage:
    python scripts/verify_and_resave_images.py
    python scripts/verify_and_resave_images.py --root datasets/SKU110K_fixed --workers 16
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "SKU110K_fixed"


def label_path_for(img_path: Path) -> Path:
    parts = list(img_path.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def process_image(img_path_str: str) -> tuple[str, str]:
    img_path = Path(img_path_str)
    label_path = label_path_for(img_path)

    img = cv2.imread(img_path_str)

    valid = (
        img is not None
        and img.ndim == 3
        and img.shape[0] > 0
        and img.shape[1] > 0
        and img.shape[2] == 3
    )

    if not valid:
        img_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
        return "deleted", img_path_str

    cv2.imwrite(img_path_str, img)
    return "resaved", img_path_str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="Dataset root containing images/ and labels/ subfolders",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of worker subprocesses")
    args = parser.parse_args()

    image_paths = sorted(str(p) for p in (args.root / "images").rglob("*.jpg"))
    print(f"Found {len(image_paths)} images under {args.root / 'images'}")

    deleted = []
    with Pool(processes=args.workers) as pool:
        for i, (status, path) in enumerate(
            pool.imap_unordered(process_image, image_paths, chunksize=16), 1
        ):
            if status == "deleted":
                deleted.append(path)
                print(f"[DELETED] {path}")
            if i % 500 == 0 or i == len(image_paths):
                print(f"Progress: {i}/{len(image_paths)}")

    print("\n--- Done ---")
    print(f"Total images checked: {len(image_paths)}")
    print(f"Re-saved (valid): {len(image_paths) - len(deleted)}")
    print(f"Deleted (invalid/corrupt): {len(deleted)}")


if __name__ == "__main__":
    main()
