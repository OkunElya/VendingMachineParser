"""
Vending Machine Grid Builder
-----------------------------
Phase 1: Click 4 corners on the vending machine image (TL, TR, BR, BL order).
Phase 2: Warped view — add/adjust rows and columns, drag lines to reposition.
Exports a YAML with normalized grid coordinates.

Usage:
    python grid_builder.py --image ./vending.jpg --output ./grid.yaml

Controls — Phase 1 (corner selection):
    Left-click       Add next corner (TL -> TR -> BR -> BL)
    Right-click      Remove last corner
    Enter / Space    Confirm corners and proceed to Phase 2
    r                Reset corners
    q                Quit

Controls — Phase 2 (grid editor):
    Left-click drag  Drag a row or column line
    +  / a           Add column
    -  / d           Remove last column
    [  / w           Add row
    ]  / s           Remove last row
    Enter / Space    Export and quit
    b                Go back to Phase 1
    q                Quit without saving
"""

import cv2
import numpy as np
import yaml
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WIN       = "Grid Builder"
WARP_W    = 800      # warped canvas width
WARP_H    = 600      # warped canvas height (adjusted to aspect ratio)

COL_BG      = (28,  28,  28)
COL_ACCENT  = (0,  210, 130)
COL_CORNER  = (60, 180, 255)
COL_HOVER   = (255, 200,  60)
COL_LINE    = (0,  210, 130)
COL_LINE_HL = (255, 200,  60)
COL_TEXT    = (220, 220, 220)
COL_DIM     = (110, 110, 110)
COL_GRID_BG = (18,  18,  40)
COL_CELL    = (40,  40,  80)

CORNER_R    = 8
SNAP_DIST   = 12     # px — how close to a line to start dragging
EDGE_MARGIN = 40     # min px from canvas edge for lines

HELP_PHASE1 = [
    "Click 4 corners: TL > TR > BR > BL",
    "Right-click: undo last corner",
    "R: reset   Enter: confirm",
]
HELP_PHASE2 = [
    "+/A  add col     -/D  del col",
    "[/W  add row     ]/S  del row",
    "Drag lines to adjust",
    "Enter: export    B: back",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def order_corners(pts):
    """Sort 4 points into [TL, TR, BR, BL]."""
    pts = np.array(pts, dtype="float32")
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype="float32")


def compute_warp_size(corners):
    """Compute output rectangle width/height from the 4 corners."""
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    return max(w, 1), max(h, 1)


def warp_image(img, corners, out_w, out_h):
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Phase1State:
    corners: list = field(default_factory=list)   # list of (x, y) tuples
    mouse:   tuple = (0, 0)


@dataclass
class Phase2State:
    warp_img:   np.ndarray = None
    # Normalised positions [0..1] for column dividers (excludes 0 and 1 edges)
    col_lines:  list = field(default_factory=list)
    row_lines:  list = field(default_factory=list)
    drag_axis:  Optional[str] = None   # 'col' or 'row'
    drag_idx:   Optional[int] = None
    mouse:      tuple = (0, 0)
    out_w:      int = 0
    out_h:      int = 0


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_text_block(canvas, lines, x, y, color=COL_DIM, scale=0.42, dy=18):
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_phase1(img_orig, state: Phase1State):
    canvas = img_orig.copy()
    h, w   = canvas.shape[:2]
    corners = state.corners

    # Draw completed edges
    for i in range(len(corners)):
        if i > 0:
            cv2.line(canvas, corners[i-1], corners[i], COL_ACCENT, 1, cv2.LINE_AA)
    if len(corners) == 4:
        cv2.line(canvas, corners[3], corners[0], COL_ACCENT, 1, cv2.LINE_AA)

    # Rubber-band line from last corner to mouse
    if 0 < len(corners) < 4:
        cv2.line(canvas, corners[-1], state.mouse, COL_DIM, 1, cv2.LINE_AA)

    # Corners
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(corners):
        cv2.circle(canvas, pt, CORNER_R, COL_CORNER, -1, cv2.LINE_AA)
        cv2.putText(canvas, labels[i], (pt[0]+10, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CORNER, 1, cv2.LINE_AA)

    # Next corner indicator
    if len(corners) < 4:
        nxt = labels[len(corners)]
        cv2.putText(canvas, f"Click: {nxt}", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ACCENT, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Press Enter to confirm", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ACCENT, 1, cv2.LINE_AA)

    # Help overlay (top-right)
    draw_text_block(canvas, HELP_PHASE1, w - 310, 20)
    return canvas


def draw_phase2(state: Phase2State):
    w, h   = state.out_w, state.out_h
    mx, my = state.mouse

    canvas = state.warp_img.copy()

    # Draw cell shading
    all_x = [0] + [int(v * w) for v in sorted(state.col_lines)] + [w]
    all_y = [0] + [int(v * h) for v in sorted(state.row_lines)] + [h]
    for ci in range(len(all_x) - 1):
        for ri in range(len(all_y) - 1):
            x0, x1 = all_x[ci], all_x[ci+1]
            y0, y1 = all_y[ri], all_y[ri+1]
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x0+1, y0+1), (x1-1, y1-1), COL_CELL, -1)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Draw column lines
    for i, v in enumerate(state.col_lines):
        px     = int(v * w)
        is_hl  = (state.drag_axis == 'col' and state.drag_idx == i) or \
                 (state.drag_axis is None and abs(mx - px) <= SNAP_DIST)
        color  = COL_LINE_HL if is_hl else COL_LINE
        thick  = 2 if is_hl else 1
        cv2.line(canvas, (px, 0), (px, h), color, thick, cv2.LINE_AA)
        cv2.putText(canvas, str(i+1), (px+3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Draw row lines
    for i, v in enumerate(state.row_lines):
        py     = int(v * h)
        is_hl  = (state.drag_axis == 'row' and state.drag_idx == i) or \
                 (state.drag_axis is None and abs(my - py) <= SNAP_DIST)
        color  = COL_LINE_HL if is_hl else COL_LINE
        thick  = 2 if is_hl else 1
        cv2.line(canvas, (0, py), (w, py), color, thick, cv2.LINE_AA)
        cv2.putText(canvas, str(i+1), (4, py-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Border
    cv2.rectangle(canvas, (0, 0), (w-1, h-1), COL_ACCENT, 1)

    # Info bar
    n_cols = len(state.col_lines) + 1
    n_rows = len(state.row_lines) + 1
    info   = f"Grid: {n_cols} cols x {n_rows} rows   |   " + \
             f"+A: add col  -D: del col  [W: add row  ]S: del row  |  Enter: export"
    cv2.putText(canvas, info, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_DIM, 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Phase 1 — corner picking
# ---------------------------------------------------------------------------

def phase1(img_orig):
    state = Phase1State()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, _):
        state.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state.corners) < 4:
                state.corners.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state.corners:
                state.corners.pop()

    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        frame = draw_phase1(img_orig, state)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            cv2.destroyAllWindows()
            sys.exit(0)

        if key == ord('r'):
            state.corners.clear()

        if key in (13, 32) and len(state.corners) == 4:
            return state.corners

    return None


# ---------------------------------------------------------------------------
# Phase 2 — grid editor
# ---------------------------------------------------------------------------

def evenly_spaced(n):
    """Return n-1 internal divider positions for n cells."""
    return [i / n for i in range(1, n)]


def phase2(img_orig, corners):
    ordered = order_corners(corners)
    ww, wh  = compute_warp_size(ordered)

    # Fit into a max display size preserving aspect
    max_dim  = 900
    scale    = min(max_dim / ww, max_dim / wh, 1.0)
    disp_w   = max(int(ww * scale), 100)
    disp_h   = max(int(wh * scale), 100)

    warped   = warp_image(img_orig, ordered, disp_w, disp_h)

    state    = Phase2State(
        warp_img  = warped,
        col_lines = evenly_spaced(4),   # default 4 cols
        row_lines = evenly_spaced(5),   # default 5 rows
        out_w     = disp_w,
        out_h     = disp_h,
    )

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, _):
        state.mouse = (x, y)
        w, h = state.out_w, state.out_h

        if event == cv2.EVENT_LBUTTONDOWN:
            # Find nearest line to start drag
            best_dist, best_axis, best_idx = SNAP_DIST + 1, None, None
            for i, v in enumerate(state.col_lines):
                d = abs(x - int(v * w))
                if d < best_dist:
                    best_dist, best_axis, best_idx = d, 'col', i
            for i, v in enumerate(state.row_lines):
                d = abs(y - int(v * h))
                if d < best_dist:
                    best_dist, best_axis, best_idx = d, 'row', i
            if best_axis:
                state.drag_axis = best_axis
                state.drag_idx  = best_idx

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.drag_axis == 'col' and state.drag_idx is not None:
                clamped = max(EDGE_MARGIN / w, min(1 - EDGE_MARGIN / w, x / w))
                state.col_lines[state.drag_idx] = clamped
            elif state.drag_axis == 'row' and state.drag_idx is not None:
                clamped = max(EDGE_MARGIN / h, min(1 - EDGE_MARGIN / h, y / h))
                state.row_lines[state.drag_idx] = clamped

        elif event == cv2.EVENT_LBUTTONUP:
            state.drag_axis = None
            state.drag_idx  = None

    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        frame = draw_phase2(state)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            cv2.destroyAllWindows()
            sys.exit(0)

        # Back to phase 1
        if key == ord('b'):
            cv2.destroyAllWindows()
            return None

        # Add / remove columns
        if key in (ord('+'), ord('a'), ord('=')):
            n = len(state.col_lines) + 2   # current cols + 1
            state.col_lines = evenly_spaced(n)

        if key in (ord('-'), ord('d')):
            if len(state.col_lines) > 0:
                n = max(1, len(state.col_lines))
                state.col_lines = evenly_spaced(n)

        # Add / remove rows
        if key in (ord('['), ord('w')):
            n = len(state.row_lines) + 2
            state.row_lines = evenly_spaced(n)

        if key in (ord(']'), ord('s')):
            if len(state.row_lines) > 0:
                n = max(1, len(state.row_lines))
                state.row_lines = evenly_spaced(n)

        # Confirm
        if key in (13, 32):
            cv2.destroyAllWindows()
            return state, ordered, ww, wh

    return None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_yaml_data(state: Phase2State, corners_ordered, orig_img_shape,
                    real_w, real_h):
    """
    corners_ordered : [TL, TR, BR, BL] in original image pixel coords
    orig_img_shape  : (H, W, C) of the original image
    real_w, real_h  : unscaled warp dimensions (for aspect ratio)
    """
    img_h, img_w = orig_img_shape[:2]

    # Normalise corners to [0..1]
    norm_corners = {
        "TL": [round(float(corners_ordered[0][0] / img_w), 6),
               round(float(corners_ordered[0][1] / img_h), 6)],
        "TR": [round(float(corners_ordered[1][0] / img_w), 6),
               round(float(corners_ordered[1][1] / img_h), 6)],
        "BR": [round(float(corners_ordered[2][0] / img_w), 6),
               round(float(corners_ordered[2][1] / img_h), 6)],
        "BL": [round(float(corners_ordered[3][0] / img_w), 6),
               round(float(corners_ordered[3][1] / img_h), 6)],
    }

    n_cols = len(state.col_lines) + 1
    n_rows = len(state.row_lines) + 1

    # All dividers including edges (0.0 and 1.0)
    col_dividers = sorted([0.0] + [round(v, 6) for v in state.col_lines] + [1.0])
    row_dividers = sorted([0.0] + [round(v, 6) for v in state.row_lines] + [1.0])

    # Cell list: each cell as {row, col, x_min, y_min, x_max, y_max} normalised in warp space
    cells = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            cells.append({
                "row":   ri,
                "col":   ci,
                "x_min": col_dividers[ci],
                "y_min": row_dividers[ri],
                "x_max": col_dividers[ci + 1],
                "y_max": row_dividers[ri + 1],
            })

    return {
        "image_size": {"width": img_w, "height": img_h},
        "perspective_corners": norm_corners,
        "warp_aspect": {
            "width":  real_w,
            "height": real_h,
        },
        "grid": {
            "cols":          n_cols,
            "rows":          n_rows,
            "col_dividers":  col_dividers,
            "row_dividers":  row_dividers,
        },
        "cells": cells,
    }


def save_output(data, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Saved: {output_path}")
    print(f"  Grid : {data['grid']['rows']} rows x {data['grid']['cols']} cols")
    print(f"  Cells: {len(data['cells'])}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vending machine grid builder — pick corners, define grid, export YAML."
    )
    parser.add_argument("--image",  "-i", required=True,
                        help="Path to the vending machine image")
    parser.add_argument("--output", "-o", default="./grid.yaml",
                        help="Output YAML path (default: ./grid.yaml)")
    parser.add_argument("--cols",   "-c", type=int, default=4,
                        help="Initial number of columns (default: 4)")
    parser.add_argument("--rows",   "-r", type=int, default=5,
                        help="Initial number of rows (default: 5)")
    args = parser.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        sys.exit(1)

    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"ERROR: Could not read image: {img_path}")
        sys.exit(1)

    print(f"Image: {img_path}  ({img_orig.shape[1]}x{img_orig.shape[0]})")
    print("Phase 1: Click 4 corners (TL -> TR -> BR -> BL), then press Enter.")

    while True:
        corners = phase1(img_orig)
        if corners is None:
            break

        result = phase2(img_orig, corners)
        if result is None:
            # User pressed B — go back to phase 1
            print("Back to corner selection.")
            continue

        state, ordered_corners, real_w, real_h = result

        # Override default grid counts if user specified
        if args.cols != 4:
            state.col_lines = evenly_spaced(args.cols)
        if args.rows != 5:
            state.row_lines = evenly_spaced(args.rows)

        data = build_yaml_data(state, ordered_corners, img_orig.shape, real_w, real_h)
        save_output(data, Path(args.output))
        break


if __name__ == "__main__":
    main()
