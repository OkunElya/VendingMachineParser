"""Synthetic occlusion augmentation that overlays randomized arcs cut from
vending-machine spring/auger PNGs (datasets/aug_springs) onto training
images, mimicking the coil dispenser that wraps across products on real
vending machine shelves.

Each spring PNG is unwrapped to polar coordinates (angle x radius) once at
load time, purely to find its "valid span" -- the angular range actually
covered by spring material (the complement of its largest gap, or the full
circle if it's a closed ring). At augmentation time, a random sub-arc within
that span is cut out with a `cv2.ellipse` sector mask applied directly to the
original (un-warped) image, so the spring's appearance is never resampled or
distorted -- only masked.
"""

from __future__ import annotations

import glob
import random
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.augment import BaseTransform

SPRINGS_DIR  = "./datasets/aug_springs"
ALPHA_THRESH = 16   # alpha values below this are treated as transparent
MIN_GAP_DEG  = 8    # gaps narrower than this are treated as noise (-> full circle)
POLAR_SIZE   = 128  # radial resolution of the polar buffer used for gap detection


def _largest_gap_span(profile: np.ndarray) -> tuple[int, int]:
    """`profile` is a 360-length bool array, True where the spring has
    material at that angle. Returns (start, length) of the spring's
    drawable arc -- the complement of the largest empty run, with circular
    wraparound. Returns (0, 360) if no empty run is >= MIN_GAP_DEG (the
    spring is a closed/full circle).
    """
    n = len(profile)
    doubled = np.concatenate([profile, profile])

    best_start, best_len, pos = 0, 0, 0
    for val, group in groupby(doubled):
        glen = sum(1 for _ in group)
        if not val and pos < n and glen > best_len:
            best_len, best_start = glen, pos
        pos += glen

    best_len = min(best_len, n)
    if best_len < MIN_GAP_DEG:
        return 0, n

    span_start = (best_start + best_len) % n
    return span_start, n - best_len


def _tight_crop(rgba: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(rgba[:, :, 3] > ALPHA_THRESH)
    if len(xs) == 0:
        return rgba[:1, :1]
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class SpringAugmenter:
    """Loads transparent-background spring/auger PNGs and generates
    randomized arc-shaped occlusion patches from them."""

    def __init__(self, springs_dir: str | Path = SPRINGS_DIR, polar_size: int = POLAR_SIZE):
        self.polar_size = polar_size
        self._springs: list[dict] = []
        for path in sorted(glob.glob(str(Path(springs_dir) / "*.png"))):
            spring = self._load_spring(path)
            if spring is not None:
                self._springs.append(spring)
        if not self._springs:
            raise FileNotFoundError(f"No usable spring PNGs found in {springs_dir!r}")

    def _load_spring(self, path: str) -> dict | None:
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            return None

        ys, xs = np.nonzero(rgba[:, :, 3] > ALPHA_THRESH)
        if len(xs) == 0:
            return None

        center = (float(xs.mean()), float(ys.mean()))
        radius = float(np.hypot(xs - center[0], ys - center[1]).max())
        if radius < 1:
            return None

        polar   = cv2.warpPolar(rgba, (self.polar_size, 360), center, radius,
                                 cv2.WARP_FILL_OUTLIERS)
        profile = polar[:, :, 3].max(axis=1) > ALPHA_THRESH
        span    = _largest_gap_span(profile)

        return {"rgba": rgba, "center": center, "radius": radius, "span": span}

    def random_patch(self, arc_range: tuple[float, float] = (180, 270)) -> np.ndarray:
        """Returns a randomized RGBA arc-shaped patch (tight-cropped to its
        own content), cut from a random spring via a sector mask -- no
        resampling, so the spring's texture is preserved as-is. `arc_range`
        bounds the arc length in degrees (clamped to the spring's own valid
        span).
        """
        spring = random.choice(self._springs)
        rgba, center, radius = spring["rgba"], spring["center"], spring["radius"]
        span_start, span_len = spring["span"]

        arc_len = int(round(min(span_len, random.uniform(*arc_range))))
        arc_len = max(1, arc_len)

        offset = random.randint(0, max(0, span_len - arc_len))
        start  = span_start + offset
        end    = start + arc_len

        h, w = rgba.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        r    = int(np.ceil(radius)) + 1
        c    = (int(round(center[0])), int(round(center[1])))
        cv2.ellipse(mask, c, (r, r), 0, start, end, 255, -1)

        out = rgba.copy()
        out[:, :, 3] = np.minimum(out[:, :, 3], mask)
        return _tight_crop(out)


def overlay_rgba(base_bgr: np.ndarray, patch_rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite `patch_rgba` onto `base_bgr` in place, with its
    top-left corner at (x, y). Parts of the patch outside `base_bgr`'s
    bounds are clipped.
    """
    h, w   = base_bgr.shape[:2]
    ph, pw = patch_rgba.shape[:2]

    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(w, x + pw), min(h, y + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return

    sx0, sy0 = dx0 - x, dy0 - y
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

    patch   = patch_rgba[sy0:sy1, sx0:sx1]
    alpha   = patch[:, :, 3:4].astype(np.float32) / 255.0
    roi     = base_bgr[dy0:dy1, dx0:dx1].astype(np.float32)
    blended = roi * (1 - alpha) + patch[:, :, :3].astype(np.float32) * alpha
    base_bgr[dy0:dy1, dx0:dx1] = blended.astype(base_bgr.dtype)


def _scale_patch(patch: np.ndarray, target_long_side: float) -> np.ndarray:
    scale = target_long_side / max(patch.shape[:2])
    new_w = max(1, int(round(patch.shape[1] * scale)))
    new_h = max(1, int(round(patch.shape[0] * scale)))
    return cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def apply_edge_overlay(image_bgr: np.ndarray, augmenter: SpringAugmenter,
                        scale_range: tuple[float, float] = (0.35, 0.75),
                        arc_range: tuple[float, float] = (180, 270),
                        overhang: tuple[float, float] = (0.05, 0.35)) -> np.ndarray:
    """For single-object crops (embedding/classifier training): overlay a
    random spring patch mostly inside the frame but crossing one edge, so
    part of the coil is clipped off -- mimicking a dispenser coil draped
    across a product whose crop is roughly the whole frame. `overhang`
    bounds the fraction of the patch's relevant dimension that's pushed
    outside that edge. Returns a copy.
    """
    out   = image_bgr.copy()
    h, w  = out.shape[:2]
    patch = _scale_patch(augmenter.random_patch(arc_range), random.uniform(*scale_range) * max(h, w))
    ph, pw = patch.shape[:2]

    edge = random.randint(0, 3)  # 0=top, 1=bottom, 2=left, 3=right
    if edge in (0, 1):
        x    = random.randint(-pw // 4, w - 3 * pw // 4)
        hang = int(random.uniform(*overhang) * ph)
        y    = -hang if edge == 0 else h - ph + hang
    else:
        y    = random.randint(-ph // 4, h - 3 * ph // 4)
        hang = int(random.uniform(*overhang) * pw)
        x    = -hang if edge == 2 else w - pw + hang

    overlay_rgba(out, patch, x, y)
    return out


class SpringOcclusionPIL:
    """torchvision.transforms.v2-compatible transform: with probability `p`,
    alpha-composites a random spring patch across one edge of a PIL image
    (see `apply_edge_overlay`). Intended for single-object training crops,
    e.g. the embedding/classifier dataloader.
    """

    def __init__(self, springs_dir: str | Path = SPRINGS_DIR, p: float = 0.3, **overlay_kwargs):
        self.augmenter = SpringAugmenter(springs_dir)
        self.p = p
        self.overlay_kwargs = overlay_kwargs

    def __call__(self, img):
        if random.random() > self.p:
            return img
        from PIL import Image
        arr = np.asarray(img)[:, :, ::-1]  # RGB -> BGR for cv2-based helpers
        arr = apply_edge_overlay(arr, self.augmenter, **self.overlay_kwargs)
        return Image.fromarray(arr[:, :, ::-1])


class SpringOcclusion(BaseTransform):
    """Ultralytics dataset transform: with probability `p`, overlays one or
    more random spring/auger arc patches anchored to randomly chosen object
    bounding boxes -- mimicking a dispenser coil draped across products on a
    shelf. Each patch is sized relative to its target bbox and jittered so it
    drapes over the box edge/corner (and often onto neighbouring objects in
    densely-packed scenes), rather than floating in empty space.

    Operates directly on `labels["img"]` (HWC BGR uint8) and
    `labels["instances"]` (bboxes only, read-only) -- intended for dense
    multi-object scene training (e.g. the YOLO OBB detector). Use
    `register_spring_occlusion` to insert it into a dataset's pipeline.

    `coverage` is the fraction of bboxes in the image that get a spring patch
    (sampled per-image), so denser scenes get proportionally more springs.
    """

    def __init__(self, springs_dir: str | Path = SPRINGS_DIR, p: float = 0.4,
                 coverage: tuple[float, float] = (0.05, 0.2),
                 scale_range: tuple[float, float] = (0.6, 1.4),
                 arc_range: tuple[float, float] = (180, 270)):
        self.augmenter = SpringAugmenter(springs_dir)
        self.p = p
        self.coverage = coverage
        self.scale_range = scale_range
        self.arc_range = arc_range

    def __call__(self, labels: dict) -> dict:
        if random.random() > self.p:
            return labels

        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return labels

        img = labels["img"]
        h, w = img.shape[:2]

        was_normalized = instances.normalized
        instances.convert_bbox("xywh")
        if was_normalized:
            instances.denormalize(w, h)
        bboxes = instances.bboxes

        frac = random.uniform(*self.coverage)
        n = min(max(1, round(len(bboxes) * frac)), len(bboxes))
        for i in np.random.choice(len(bboxes), size=n, replace=False):
            cx, cy, bw, bh = bboxes[i]
            patch = _scale_patch(self.augmenter.random_patch(self.arc_range),
                                  random.uniform(*self.scale_range) * max(bw, bh))
            ph, pw = patch.shape[:2]
            x = int(cx - pw / 2 + random.uniform(-0.3, 0.3) * bw)
            y = int(cy - ph / 2 + random.uniform(-0.3, 0.3) * bh)
            overlay_rgba(img, patch, x, y)

        if was_normalized:
            instances.normalize(w, h)
        return labels


def register_spring_occlusion(dataset, **kwargs) -> SpringOcclusion:
    """Insert a `SpringOcclusion` transform into `dataset`'s pipeline, right
    before the final `Format` transform. Also wraps `dataset.build_transforms`
    so the occlusion survives `close_mosaic` rebuilds, which call
    `build_transforms` again partway through training.

    Intended for use from an `on_pretrain_routine_end` callback, e.g.:
        model.add_callback("on_pretrain_routine_end",
                            lambda trainer: register_spring_occlusion(trainer.train_loader.dataset))
    or directly on a dataset built via `build_yolo_dataset` for previewing.
    """
    spring_occlusion = SpringOcclusion(**kwargs)
    base_build_transforms = dataset.build_transforms

    def build_transforms(hyp=None):
        transforms = base_build_transforms(hyp)
        transforms.insert(-1, spring_occlusion)
        return transforms

    dataset.build_transforms = build_transforms
    dataset.transforms.insert(-1, spring_occlusion)
    return spring_occlusion
