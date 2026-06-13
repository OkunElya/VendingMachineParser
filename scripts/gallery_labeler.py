"""
Gallery Labeler
---------------
Interactive tool for building/extending the product embedding gallery
(`./gallery/<class_name>/*.png`) used by `ProductBank`.

Runs the full detection pipeline (`Pipeline.detect`, with `classify=False`)
over every image in an input directory: machine detection -> machine
classification -> window segmentation -> item detection masked to the
rectified window. This mirrors exactly what the live API does up to (but
not including) product recognition, which is what this tool is used to
build the gallery for. For each detected item it shows the crop next to a
merged suggestions/search list:
with no search text it shows the 5 closest existing gallery classes (cosine
similarity, as a percentage); once you start typing it switches to a fuzzy
search over class names (with a "+ New class" option to create one).

Usage:
    python scripts/gallery_labeler.py --input ./workspace/some_photos
    python scripts/gallery_labeler.py --input ./workspace/some_photos --gallery ./gallery

Controls:
    Ctrl+Y           Accept the #1 suggested class for the selected item
    type text        Fuzzy-search existing class names / name a new class
    Up / Down        Move the highlighted entry in the suggestions/search list
    Tab              Accept the highlighted entry (creates a new class if
                     it's the "+ New class" entry)
    Enter            Classify into the typed class name (created if new)
    Backspace        Remove last typed character
    Ctrl+D           Discard the selected item (no class assigned)
    Ctrl+Left/Right  Move the selection to the previous/next item
    Left arrow       Next image
    Right arrow      Previous image
    Esc              Clear search text, or quit if search is already empty

Embedding updates:
    - Creating a brand new class immediately (re)computes its embedding.
    - Every 10 classified items, embeddings for all classes that received
      new samples since the last update are recomputed (each from its own
      gallery images only -- other classes are left untouched).
    - Both run in a background thread; progress is shown in the bottom bar.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "python"))

from grid_helper import crop_obb_rotated, draw_obb
from pipeline import Pipeline
from shared import GALLERY_DIR, IMAGE_EXTS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRAME_SIZE    = 600     # cropped-item preview, fixed per spec
PANEL_W       = 440     # right-hand info panel width
HEADER_H      = 40
BOTTOM_H      = 32
UPDATE_EVERY  = 10

WINDOW_NAME = "Gallery Labeler"

# Light theme palette (RGB)
BG           = (240, 240, 240)
PANEL_BG     = (250, 250, 250)
HEADER_BG    = (203, 219, 235)
LINE_COLOR   = (210, 210, 210)
TEXT         = (40, 40, 40)
DIM          = (150, 150, 150)
ACCENT       = (20, 110, 200)
HIGHLIGHT_BG = (190, 222, 245)
INPUT_BG     = (255, 255, 255)
PAD_GRAY     = (170, 170, 170)

# Bounding-box colors (RGB; reversed to BGR when drawn with cv2)
COLOR_UNTOUCHED = (140, 30, 50)
COLOR_SELECTED  = (255, 0, 60)    # vibrant red/magenta -- drawn after the
                                   # pane is resized so it stays crisp instead
                                   # of fading toward gray when downscaled
COLOR_DONE      = (40, 160, 40)

FONT       = ImageFont.load_default(size=15)
FONT_SMALL = ImageFont.load_default(size=12)


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ItemEntry:
    obb:         np.ndarray
    state:       str = "pending"          # "pending" | "done"
    suggestions: list | None = None       # cached top-k [(name, score)]


@dataclass
class ImageEntry:
    path:     Path
    image:    np.ndarray | None = None
    items:    list = field(default_factory=list)
    selected: int = -1
    detected: bool = False


# ---------------------------------------------------------------------------
# Detection / cropping helpers
# ---------------------------------------------------------------------------

def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def order_reading(obbs: np.ndarray, image_h: int) -> list[int]:
    """Index order roughly top-to-bottom rows, left-to-right within a row."""
    if len(obbs) == 0:
        return []
    row_h = max(image_h * 0.05, 1.0)
    return sorted(range(len(obbs)), key=lambda i: (round(float(obbs[i][1]) / row_h), float(obbs[i][0])))


def crop_from_obb(image: np.ndarray, obb: np.ndarray) -> np.ndarray | None:
    crop = crop_obb_rotated(image, obb)
    return crop if crop.size > 0 else None


# ---------------------------------------------------------------------------
# Image-fitting helpers (operate on BGR numpy, like the rest of the codebase)
# ---------------------------------------------------------------------------

def fit_to_box(img: np.ndarray, box_w: int, box_h: int, bg=BG) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Scale (up or down) to fit inside box_w x box_h, centred on `bg`.

    Returns (canvas, scale, (x_offset, y_offset)) so callers can map
    coordinates from the original image onto the returned canvas
    (e.g. to draw OBBs after resizing instead of before, which keeps thin
    boxes crisp instead of fading toward gray when heavily downscaled).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((box_h, box_w, 3), bg, dtype=np.uint8), 1.0, (0, 0)
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas  = np.full((box_h, box_w, 3), bg, dtype=np.uint8)
    y0, x0  = (box_h - new_h) // 2, (box_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, scale, (x0, y0)


def scale_obb(obb: np.ndarray, scale: float, offset: tuple[int, int]) -> np.ndarray:
    """Map an xywhr OBB from original-image space onto a fit_to_box canvas."""
    x, y, w, h, r = obb
    ox, oy = offset
    return np.array([x * scale + ox, y * scale + oy, w * scale, h * scale, r], dtype=np.float32)


def make_frame(crop: np.ndarray | None, size: int = FRAME_SIZE, pad=PAD_GRAY) -> np.ndarray:
    """Pad to square (gray) to fix aspect ratio, then resize to size x size."""
    if crop is None or crop.size == 0:
        return np.full((size, size, 3), pad, dtype=np.uint8)
    h, w  = crop.shape[:2]
    side  = max(h, w)
    pad_t = (side - h) // 2
    pad_b = side - h - pad_t
    pad_l = (side - w) // 2
    pad_r = side - w - pad_l
    sq = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=pad)
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(sq, (size, size), interpolation=interp)


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Fuzzy search
# ---------------------------------------------------------------------------

def fuzzy_search(query: str, choices: list[str], limit: int = 5) -> list[str]:
    if not query:
        return list(choices[:limit])
    if not choices:
        return []
    results = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)
    return [name for name, _score, _idx in results]


def safe_dirname(name: str) -> str:
    cleaned = "".join(c for c in name if c not in '\\/:*?"<>|').strip()
    return cleaned or "unnamed"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class App:
    def __init__(self, image_paths: list[Path], pipeline: Pipeline, gallery_dir: Path):
        self.pipeline     = pipeline
        self.product_bank = pipeline.product_bank
        self.gallery_dir  = gallery_dir
        # Used only by ProductBank.recompute_class/build_cache to crop the
        # largest item out of already-cropped gallery images -- the same
        # detector call the live pipeline uses for product recognition.
        self.item_detector_fn = lambda img: pipeline._detect_items(img).cpu().numpy().astype(np.float32)

        self.entries = [ImageEntry(path=p) for p in image_paths]
        self.img_idx = 0

        self.search_text        = ""
        self.highlighted         = 0
        self.dirty_classes       = set()
        self.items_since_update  = 0
        self.status_msg          = ""

        # background embedding-update worker
        self._classes_to_update: set[str] = set()
        self._classes_lock = threading.Lock()
        self._update_running = False
        self._update_queue: queue.Queue = queue.Queue()
        self.update_active   = False
        self.update_progress = 0.0
        self.update_status_text = "Embeddings up to date"

        self.root = tk.Tk()
        self.root.title(WINDOW_NAME)
        self.root.configure(bg=rgb_hex(BG))
        self._compute_layout()
        self._build_widgets()

        self.root.bind("<KeyPress>", self.on_key)
        self._check_gallery_sync()
        self._queue_missing_embeddings()
        self.render_main()
        self.root.after(150, self.poll_updates)

    # -- layout / widgets ------------------------------------------------------------

    def _compute_layout(self) -> None:
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        self.frame_size = FRAME_SIZE
        self.panel_w    = PANEL_W
        self.header_h   = HEADER_H

        avail_h = sh - 140  # leave room for window chrome / taskbar / bottom bar
        self.pane_size = max(self.frame_size, min(1080, avail_h - self.header_h))

        max_w   = sw - 60
        total_w = self.pane_size + self.frame_size + self.panel_w
        if total_w > max_w:
            self.pane_size = max(self.frame_size, max_w - self.frame_size - self.panel_w)

        self.canvas_w = self.pane_size + self.frame_size + self.panel_w
        self.canvas_h = self.header_h + self.pane_size

    def _build_widgets(self) -> None:
        self.image_label = tk.Label(self.root, bg=rgb_hex(BG), bd=0)
        self.image_label.pack()

        bottom = tk.Frame(self.root, bg=rgb_hex(PANEL_BG), height=BOTTOM_H)
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, orient="horizontal", mode="determinate",
                                         length=260, maximum=1.0)
        self.progress.pack(side="left", padx=10, pady=5)
        self.update_label = tk.Label(bottom, text=self.update_status_text,
                                      bg=rgb_hex(PANEL_BG), fg=rgb_hex(TEXT), anchor="w")
        self.update_label.pack(side="left", fill="x", expand=True, padx=8)

    # -- detection / suggestions -------------------------------------------------

    def ensure_detected(self, entry: ImageEntry) -> None:
        if entry.detected:
            return
        entry.detected = True
        img = cv2.imread(str(entry.path))
        if img is None:
            entry.image = None
            entry.items = []
            return

        detections = self.pipeline.detect(img, classify=False)
        if not detections:
            entry.image    = img
            entry.items    = []
            entry.selected = -1
            return

        det = detections[0]
        entry.image = det.image  # machine_bb_img: cropped to the detected machine
        obbs  = [item["obb"] for item in det.items]
        order = order_reading(obbs, det.image.shape[0])
        entry.items    = [ItemEntry(obb=obbs[i]) for i in order]
        entry.selected = 0 if entry.items else -1
        if len(detections) > 1:
            self.status_msg = f"Note: {len(detections)} machines detected, showing #1"

    def get_suggestions(self, item: ItemEntry, crop: np.ndarray | None = None) -> list[tuple[str, float]]:
        if item.suggestions is None:
            if crop is None:
                entry = self.entries[self.img_idx]
                crop  = crop_from_obb(entry.image, item.obb)
            item.suggestions = self.product_bank.lookup_topk(crop, k=5) if crop is not None else []
        return item.suggestions

    def current_suggestions(self) -> list[tuple[str, float]]:
        entry = self.entries[self.img_idx]
        if entry.selected < 0 or not entry.items:
            return []
        return self.get_suggestions(entry.items[entry.selected])

    def invalidate_pending_suggestions(self) -> None:
        for entry in self.entries:
            for item in entry.items:
                if item.state == "pending":
                    item.suggestions = None

    # -- merged suggestions / fuzzy-search list --------------------------------------

    def fuzzy_matches(self) -> list[str]:
        text  = self.search_text.strip()
        names = self.product_bank.class_names
        if not text:
            return names[:5]
        matches = fuzzy_search(text, names, limit=4)
        exact   = any(n.lower() == text.lower() for n in names)
        if not exact:
            matches = matches + [f'+ New class: "{text}"']
        return matches

    def get_display_list(self) -> list[tuple[str, str]]:
        """[(name, percent_or_empty)] -- search results if typing, else suggestions."""
        if self.search_text:
            return [(m, "") for m in self.fuzzy_matches()]
        return [(name, f"{score * 100:.0f}%") for name, score in self.current_suggestions()]

    # -- navigation ----------------------------------------------------------------

    def advance_selection(self, entry: ImageEntry) -> None:
        n = len(entry.items)
        if n == 0:
            entry.selected = -1
            return
        for offset in range(1, n + 1):
            idx = (entry.selected + offset) % n
            if entry.items[idx].state == "pending":
                entry.selected = idx
                return
        entry.selected = -1

    def move_selection(self, delta: int) -> None:
        entry = self.entries[self.img_idx]
        self.ensure_detected(entry)
        n = len(entry.items)
        if n == 0:
            self.status_msg = "No items detected."
            return
        entry.selected = 0 if entry.selected < 0 else (entry.selected + delta) % n
        self.search_text = ""
        self.highlighted  = 0
        self.status_msg   = ""

    def go_to_image(self, delta: int) -> None:
        new_idx = self.img_idx + delta
        if 0 <= new_idx < len(self.entries):
            self.img_idx     = new_idx
            self.search_text = ""
            self.highlighted  = 0
            self.status_msg   = ""
        else:
            self.status_msg = "No more images" if delta > 0 else "Already at first image"

    # -- classification actions ------------------------------------------------------

    def save_crop(self, class_name: str, crop: np.ndarray, source: Path, item_idx: int) -> Path:
        class_dir = self.gallery_dir / safe_dirname(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        base = f"{source.stem}_{item_idx:03d}"
        dest = class_dir / f"{base}.png"
        n = 1
        while dest.exists():
            dest = class_dir / f"{base}_{n}.png"
            n += 1
        cv2.imwrite(str(dest), crop)
        return dest

    def _confirm_not_outlier(self, class_name: str, crop: np.ndarray) -> bool:
        """If `crop`'s embedding is unusually far from `class_name`'s
        existing gallery mean (more than 2x the typical intra-class spread
        across the gallery), ask the user to confirm before saving it as a
        mistake-prevention measure. Returns False if the user cancels."""
        global_mean = self.product_bank.global_mean_intra_class_distance
        if global_mean is None or global_mean <= 0:
            return True
        emb  = self.product_bank.embed(crop)
        dist = self.product_bank.class_mean_distance(class_name, emb)
        if dist is None or dist <= 2 * global_mean:
            return True
        return messagebox.askyesno(
            "Possible mismatch",
            f"This item looks unusually different from existing '{class_name}' "
            f"samples (distance {dist:.3f} vs typical {global_mean:.3f}).\n\n"
            f"Add it to '{class_name}' anyway?",
        )

    def classify_current(self, class_name: str) -> None:
        class_name = class_name.strip()
        if not class_name:
            self.status_msg = "Empty class name, ignored."
            return

        entry = self.entries[self.img_idx]
        if entry.selected < 0 or not entry.items:
            self.status_msg = "Nothing to classify."
            return

        item = entry.items[entry.selected]
        crop = crop_from_obb(entry.image, item.obb)
        if crop is None:
            item.state = "done"
            self.status_msg = "Empty crop, item skipped."
            self.advance_selection(entry)
            return

        if not self._confirm_not_outlier(class_name, crop):
            self.status_msg = "Cancelled - item left pending."
            return

        item.state = "done"
        is_new = class_name not in self.product_bank.class_names
        dest   = self.save_crop(class_name, crop, entry.path, entry.selected)

        if is_new:
            self.request_embedding_update({class_name})
            self.dirty_classes.discard(class_name)
            self.status_msg = f"New class '{class_name}' created -> {dest.name}"
        else:
            self.dirty_classes.add(class_name)
            self.items_since_update += 1
            self.status_msg = f"-> {class_name}  ({dest.name})"
            if self.items_since_update >= UPDATE_EVERY:
                self.request_embedding_update(self.dirty_classes)
                self.dirty_classes = set()
                self.items_since_update = 0

        self.search_text = ""
        self.highlighted  = 0
        self.advance_selection(entry)

    def accept_top_suggestion(self) -> None:
        entry = self.entries[self.img_idx]
        if entry.selected < 0 or not entry.items:
            self.status_msg = "Nothing to classify."
            return
        suggestions = self.current_suggestions()
        if not suggestions:
            self.status_msg = "No suggestions available."
            return
        self.classify_current(suggestions[0][0])

    def accept_highlighted(self) -> None:
        """Tab: accept the highlighted suggestion / fuzzy-search match."""
        display = self.get_display_list()
        if not display or not (0 <= self.highlighted < len(display)):
            self.status_msg = "Nothing selected."
            return
        name, _ = display[self.highlighted]
        if name.startswith("+ New class:"):
            self.classify_current(self.search_text)
        else:
            self.classify_current(name)

    def create_new_class(self) -> None:
        """Enter: classify into the typed class name (created if it doesn't exist)."""
        name = self.search_text.strip()
        if not name:
            self.status_msg = "Type a class name, then press Enter."
            return
        self.classify_current(name)

    def discard_current(self) -> None:
        entry = self.entries[self.img_idx]
        if entry.selected < 0 or not entry.items:
            self.status_msg = "Nothing to discard."
            return
        entry.items[entry.selected].state = "done"
        self.status_msg = f"Discarded item {entry.selected + 1}"
        self.advance_selection(entry)

    # -- background embedding updates ------------------------------------------------

    def _check_gallery_sync(self) -> None:
        """Print a warning at startup if the embedding count stored in the
        gallery .npy doesn't match the number of class folders on disk."""
        if not self.gallery_dir.is_dir():
            return
        folder_classes   = {d.name for d in self.gallery_dir.iterdir() if d.is_dir()}
        embedded_classes = set(self.product_bank.class_names)
        if len(folder_classes) == len(embedded_classes):
            return

        print(f"[gallery] WARNING: {len(embedded_classes)} embedded classes "
              f"!= {len(folder_classes)} gallery folders")
        missing  = folder_classes - embedded_classes
        orphaned = embedded_classes - folder_classes
        if missing:
            print(f"  missing embeddings ({len(missing)}): {sorted(missing)}")
        if orphaned:
            print(f"  orphaned embeddings ({len(orphaned)}): {sorted(orphaned)}")
            print(f"  deleting stale gallery cache: {self.product_bank.gallery_path}")
            self.product_bank.clear_gallery()

    def _queue_missing_embeddings(self) -> None:
        """On boot, (re)compute embeddings for any gallery class folder that
        doesn't have one yet (e.g. added by hand between runs), or that's
        missing intra-class spread stats (used by the outlier check below).
        No-op if the gallery directory is empty/missing or already up to
        date."""
        if not self.gallery_dir.is_dir():
            return
        folders    = {d.name for d in self.gallery_dir.iterdir() if d.is_dir()}
        existing   = set(self.product_bank.class_names)
        has_spread = self.product_bank.spread_class_names
        missing    = (folders - existing) | ((folders & existing) - has_spread)
        self.request_embedding_update(missing)

    def request_embedding_update(self, class_names: set[str]) -> None:
        if not class_names:
            return
        with self._classes_lock:
            self._classes_to_update.update(class_names)
            already_running = self._update_running
            self._update_running = True
        if not already_running:
            threading.Thread(target=self._embedding_worker, daemon=True).start()

    def _embedding_worker(self) -> None:
        while True:
            with self._classes_lock:
                if not self._classes_to_update:
                    self._update_running = False
                    break
                name = self._classes_to_update.pop()

            def cb(class_name, done, total, name=name):
                self._update_queue.put(("progress", name, done, total))

            ok = self.product_bank.recompute_class(
                name, self.item_detector_fn, gallery_dir=str(self.gallery_dir), progress_cb=cb)
            self._update_queue.put(("class_done", name, ok))
        self._update_queue.put(("all_done",))

    def poll_updates(self) -> None:
        finished = False
        try:
            while True:
                msg = self._update_queue.get_nowait()
                if msg[0] == "progress":
                    _, name, done, total = msg
                    self.update_active = True
                    self.update_status_text = f"Updating '{name}': {done}/{total}"
                    self.update_progress = done / total if total else 1.0
                elif msg[0] == "class_done":
                    _, name, ok = msg
                    self.update_status_text = f"'{name}' updated" if ok else f"'{name}' skipped (no images)"
                elif msg[0] == "all_done":
                    self.update_active = False
                    self.update_status_text = "Embeddings up to date"
                    self.update_progress = 1.0
                    finished = True
        except queue.Empty:
            pass

        self.progress["value"] = self.update_progress
        self.update_label.configure(text=self.update_status_text)

        if finished:
            self.invalidate_pending_suggestions()
            self.render_main()

        self.root.after(150, self.poll_updates)

    # -- rendering ----------------------------------------------------------------

    def render_main(self) -> None:
        entry = self.entries[self.img_idx]
        self.ensure_detected(entry)

        canvas = Image.new("RGB", (self.canvas_w, self.canvas_h), BG)
        draw   = ImageDraw.Draw(canvas)

        # -- header --
        draw.rectangle([0, 0, self.canvas_w, self.header_h], fill=HEADER_BG)
        item_str = f"{entry.selected + 1}/{len(entry.items)}" if entry.items else "-/0"
        header = (f"Image {self.img_idx + 1}/{len(self.entries)}: {entry.path.name}   |   "
                  f"Item {item_str}   |   Classes: {len(self.product_bank.class_names)}")
        draw.text((12, 12), header, fill=TEXT, font=FONT)

        # -- main image pane --
        if entry.image is not None:
            pane_canvas, scale, offset = fit_to_box(entry.image, self.pane_size, self.pane_size)
            for i, item in enumerate(entry.items):
                if i == entry.selected:
                    color, thick = COLOR_SELECTED, 3
                elif item.state == "done":
                    color, thick = COLOR_DONE, 2
                else:
                    color, thick = COLOR_UNTOUCHED, 2
                draw_obb(pane_canvas, scale_obb(item.obb, scale, offset), color=color[::-1], thickness=thick)
            pane_img = bgr_to_pil(pane_canvas)
        else:
            pane_img = Image.new("RGB", (self.pane_size, self.pane_size), BG)
            ImageDraw.Draw(pane_img).text((20, self.pane_size // 2), "Failed to load image",
                                           fill=TEXT, font=FONT)
        canvas.paste(pane_img, (0, self.header_h))

        # -- frame pane (cropped item) --
        crop = None
        if entry.selected >= 0 and entry.items:
            item = entry.items[entry.selected]
            crop = crop_from_obb(entry.image, item.obb)
            frame_img = bgr_to_pil(make_frame(crop))
            suggestions = self.get_suggestions(item, crop)
        else:
            frame_img = Image.new("RGB", (self.frame_size, self.frame_size), PAD_GRAY)
            msg = "All items done" if entry.items else "No items detected"
            ImageDraw.Draw(frame_img).text((20, self.frame_size // 2), msg, fill=TEXT, font=FONT)
            suggestions = []
        fy = self.header_h + (self.pane_size - self.frame_size) // 2
        canvas.paste(frame_img, (self.pane_size, fy))

        # -- side panel --
        canvas.paste(self.render_panel(), (self.pane_size + self.frame_size, self.header_h))

        self.photo = ImageTk.PhotoImage(canvas)
        self.image_label.configure(image=self.photo)

    def render_panel(self) -> Image.Image:
        panel = Image.new("RGB", (self.panel_w, self.pane_size), PANEL_BG)
        draw  = ImageDraw.Draw(panel)
        draw.line([(0, 0), (0, self.pane_size)], fill=LINE_COLOR, width=1)
        y = 14

        display = self.get_display_list()
        in_search = bool(self.search_text)
        title = "SEARCH RESULTS" if in_search else "SUGGESTIONS  (Ctrl+Y = accept #1)"
        draw.text((14, y), title, fill=ACCENT, font=FONT)
        y += 26

        if not display:
            msg = "(type a name to create the first class)" if in_search else "(gallery is empty)"
            draw.text((14, y), msg, fill=DIM, font=FONT_SMALL)
            y += 24
        for i, (name, pct) in enumerate(display):
            row_h = 26
            if i == self.highlighted:
                draw.rectangle([8, y - 3, self.panel_w - 8, y + row_h - 6], fill=HIGHLIGHT_BG)
            color = ACCENT if name.startswith("+ New class:") else TEXT
            draw.text((14, y), f"{i + 1}. {name}", fill=color, font=FONT)
            if pct:
                w = draw.textlength(pct, font=FONT)
                draw.text((self.panel_w - 14 - w, y), pct, fill=DIM, font=FONT)
            y += row_h
        y += 10

        draw.line([(8, y), (self.panel_w - 8, y)], fill=LINE_COLOR, width=1)
        y += 20

        # -- search box --
        draw.text((14, y), "SEARCH / NEW CLASS", fill=ACCENT, font=FONT)
        y += 24
        draw.rectangle([8, y - 3, self.panel_w - 8, y + 23], fill=INPUT_BG, outline=LINE_COLOR)
        draw.text((14, y), self.search_text + "_", fill=TEXT, font=FONT)
        y += 38

        draw.line([(8, y), (self.panel_w - 8, y)], fill=LINE_COLOR, width=1)
        y += 20

        # -- hotkeys --
        draw.text((14, y), "HOTKEYS", fill=ACCENT, font=FONT)
        y += 24
        for key_str, desc in [
            ("Ctrl+Y",          "Accept top suggestion"),
            ("type",            "Search / new class name"),
            ("Up / Down",       "Move highlighted match"),
            ("Tab",             "Accept highlighted match"),
            ("Enter",           "Create class from typed name"),
            ("Backspace",       "Delete character"),
            ("Ctrl+D",          "Discard item"),
            ("Ctrl+Left/Right", "Prev / next item"),
            ("Left / Right",    "Next / previous image"),
            ("Esc",             "Clear search / quit"),
        ]:
            draw.text((14, y), key_str, fill=ACCENT, font=FONT_SMALL)
            draw.text((175, y), desc, fill=DIM, font=FONT_SMALL)
            y += 20

        if self.status_msg:
            draw.text((14, self.pane_size - 24), self.status_msg, fill=ACCENT, font=FONT_SMALL)

        return panel

    # -- key handling ----------------------------------------------------------------

    def on_key(self, event: tk.Event) -> None:
        ctrl   = bool(event.state & 0x4)
        keysym = event.keysym

        if ctrl and keysym == "Left":
            self.move_selection(-1)
        elif ctrl and keysym == "Right":
            self.move_selection(+1)
        elif ctrl and keysym in ("y", "Y"):
            self.accept_top_suggestion()
        elif ctrl and keysym in ("d", "D"):
            self.discard_current()
        elif not ctrl and keysym == "Left":
            self.go_to_image(-1)
        elif not ctrl and keysym == "Right":
            self.go_to_image(+1)
        elif keysym == "Up":
            n = len(self.get_display_list())
            if n:
                self.highlighted = (self.highlighted - 1) % n
        elif keysym == "Down":
            n = len(self.get_display_list())
            if n:
                self.highlighted = (self.highlighted + 1) % n
        elif keysym == "Return":
            self.create_new_class()
        elif keysym == "Tab":
            self.accept_highlighted()
        elif keysym == "BackSpace":
            self.search_text = self.search_text[:-1]
            self.highlighted = 0
        elif keysym == "Escape":
            if self.search_text:
                self.search_text = ""
                self.highlighted = 0
            else:
                self.root.destroy()
                return
        elif not ctrl and len(event.char) == 1 and 32 <= ord(event.char) <= 126:
            self.search_text += event.char
            self.highlighted = 0

        self.render_main()
        return "break"

    # -- main loop ----------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive item-detection gallery builder.")
    parser.add_argument("--input", "-i", required=True, help="Directory of images to process")
    parser.add_argument("--gallery", "-g", default=GALLERY_DIR, help=f"Gallery directory (default: {GALLERY_DIR})")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")

    images = collect_images(input_dir)
    if not images:
        sys.exit(f"No images found in {input_dir}")

    print(f"{len(images)} images found in {input_dir}")
    print("Loading models...")
    pipeline    = Pipeline()
    gallery_dir = Path(args.gallery).expanduser().resolve()
    gallery_dir.mkdir(parents=True, exist_ok=True)

    App(images, pipeline, gallery_dir).run()


if __name__ == "__main__":
    main()
