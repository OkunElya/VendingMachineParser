from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any

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


def wrap_points(points, corners, out_w, out_h):
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    pts = points.astype("float32").reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def center_of_obb(obb):
    """Return the (x, y) center of an OBB given in xywhr format."""
    return np.array([float(obb[0]), float(obb[1])])


# ---------------------------------------------------------------------------
# OBB drawing helpers
# ---------------------------------------------------------------------------

def obb_xywhr_to_corners(obb):
    """Convert xywhr OBB to 4 corner points (float32, shape 4x2)."""
    x, y, w, h, r = float(obb[0]), float(obb[1]), float(obb[2]), float(obb[3]), float(obb[4])
    cos_r, sin_r = np.cos(r), np.sin(r)
    hw, hh = w / 2, h / 2
    dx = np.array([-hw,  hw,  hw, -hw])
    dy = np.array([-hh, -hh,  hh,  hh])
    cx = x + dx * cos_r - dy * sin_r
    cy = y + dx * sin_r + dy * cos_r
    return np.stack([cx, cy], axis=1).astype(np.float32)


def draw_obb(image, obb, color=(0, 255, 0), thickness=2):
    """Draw a single OBB (xywhr) on image in-place."""
    corners = obb_xywhr_to_corners(obb).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [corners], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    return image


def draw_obbs(image, obbs, color=(0, 255, 0), thickness=2):
    """Draw multiple OBBs (each in xywhr format) on image in-place."""
    for obb in obbs:
        draw_obb(image, obb, color=color, thickness=thickness)
    return image


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GridCell:
    row:           int
    col:           int
    col_span:      int
    obb:           Any            # xywhr tensor from YOLO
    center_warped: np.ndarray
    left_x:        float
    right_x:       float
    top_y:         float
    bottom_y:      float
    product_name:  str | None = None
    product_score: float      = 0.0


@dataclass
class GridResult:
    n_rows:          int
    n_cols:          int
    n_cols_per_row:  list
    row_boundaries:  list
    row_col_borders: list
    out_w:           int
    out_h:           int
    cells:           list[GridCell]
    grid_2d:         list


# ---------------------------------------------------------------------------
# Grid fitting
# ---------------------------------------------------------------------------

def build_grid(machine_info, window_points, items) -> GridResult | None:
    """
    Build a perspective-corrected grid from detected OBBs.

    Rows: items whose warped bottom-Y values are within 3% of the y-span
          are merged into one row.
    Cols: unit width = mean of the narrowest 25% of items; each item occupies
          floor(item_width / unit) slots, min 1. Columns are assigned
          left-to-right per row using actual item left/right x coordinates.

    Returns None when items list is empty.
    """
    if not items:
        return None

    ordered      = order_corners(window_points)
    out_w, out_h = compute_warp_size(ordered)

    N = len(items)

    corners_orig = np.array([obb_xywhr_to_corners(item["obb"]) for item in items])  # (N,4,2)
    warped_all   = wrap_points(corners_orig.reshape(-1, 2), ordered, out_w, out_h).reshape(N, 4, 2)

    warped_centers = warped_all.mean(axis=1)
    bottom_ys      = warped_all[:, :, 1].max(axis=1)
    top_ys         = warped_all[:, :, 1].min(axis=1)
    left_xs        = warped_all[:, :, 0].min(axis=1)
    right_xs       = warped_all[:, :, 0].max(axis=1)
    warped_widths  = right_xs - left_xs

    # --- Row detection -------------------------------------------------------
    y_span    = float(bottom_ys.max() - bottom_ys.min())
    threshold = 0.03 * y_span if y_span > 1.0 else 1.0

    sorted_by_y = list(np.argsort(bottom_ys))
    row_groups: list[list[int]] = []
    cur_group = [sorted_by_y[0]]

    for idx in sorted_by_y[1:]:
        if abs(bottom_ys[idx] - bottom_ys[cur_group[0]]) <= threshold:
            cur_group.append(idx)
        else:
            row_groups.append(cur_group)
            cur_group = [idx]
    row_groups.append(cur_group)

    n_rows = len(row_groups)
    row_of = np.zeros(N, dtype=int)
    for r, grp in enumerate(row_groups):
        for i in grp:
            row_of[i] = r

    # --- Column detection ----------------------------------------------------
    max_cols    = machine_info.get("max_cols", 10)
    bottom_25_n = max(1, N // 4)
    unit_width  = float(np.sort(warped_widths)[:bottom_25_n].mean())
    col_spans   = np.maximum(1, np.floor((warped_widths / unit_width)+0.25).astype(int))
    col_of:          np.ndarray       = np.zeros(N, dtype=int)
    n_cols_per_row:  list[int]        = []
    row_col_borders: list[list[float]] = []

    for grp in row_groups:
        row_sorted = sorted(grp, key=lambda i: left_xs[i])
        borders: list[float] = []
        col = 0
        last_right = min(left_xs) #imagnary item at the beggining of each row 
        for i in row_sorted:
            if (left_xs[i]-last_right) > 0.5 * unit_width :
                borders.append(float(right_xs[i]))
                col+=1
            borders.append(float(left_xs[i]))
            col_of[i] = col
            col += int(col_spans[i])
            last_right = right_xs[i]
        borders.append(float(right_xs[row_sorted[-1]]))
        row_col_borders.append(borders)
        n_cols_per_row.append(col)

    n_cols = min(max_cols, max(n_cols_per_row))

    # --- Row boundary Y values -----------------------------------------------
    row_boundaries: list[float] = []
    for grp in row_groups:
        row_boundaries.append(float(top_ys[grp].min()))
    row_boundaries.append(float(bottom_ys[row_groups[-1]].max()))

    # --- Build GridCell objects -----------------------------------------------
    cells = [
        GridCell(
            row=int(row_of[i]),
            col=int(col_of[i]),
            col_span=int(col_spans[i]),
            obb=items[i]["obb"],
            center_warped=warped_centers[i],
            left_x=float(left_xs[i]),
            right_x=float(right_xs[i]),
            top_y=float(top_ys[i]),
            bottom_y=float(bottom_ys[i]),
        )
        for i in range(N)
    ]

    # --- 2-D grid allocation -------------------------------------------------
    grid_2d: list[list[int | None]] = [[None] * n_cols for _ in range(n_rows)]
    for idx, cell in enumerate(cells):
        for dc in range(cell.col_span):
            if cell.col + dc < n_cols:
                grid_2d[cell.row][cell.col + dc] = idx

    return GridResult(
        n_rows=n_rows,
        n_cols=n_cols,
        n_cols_per_row=n_cols_per_row,
        row_boundaries=row_boundaries,
        row_col_borders=row_col_borders,
        out_w=out_w,
        out_h=out_h,
        cells=cells,
        grid_2d=grid_2d,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_grid(machine_image: np.ndarray,
                   window_points: np.ndarray,
                   grid: GridResult) -> None:
    """Show the rectified window with row/column grid lines and cell labels."""
    ordered     = order_corners(window_points)
    warped_view = warp_image(machine_image, ordered, grid.out_w, grid.out_h)

    for y in grid.row_boundaries:
        cv2.line(warped_view, (0, int(y)), (grid.out_w, int(y)),
                 (0, 200, 255), 1, cv2.LINE_AA)

    for r, borders in enumerate(grid.row_col_borders):
        y_top = int(grid.row_boundaries[r])
        y_bot = int(grid.row_boundaries[r + 1])
        for x in borders:
            cv2.line(warped_view, (int(x), y_top), (int(x), y_bot),
                     (0, 200, 255), 1, cv2.LINE_AA)

    for cell in grid.cells:
        cx, cy = int(cell.center_warped[0]), int(cell.center_warped[1])
        cv2.circle(warped_view, (cx, cy), 5, (0, 0, 255), -1)
        label = f"{cell.row},{cell.col}"
        if cell.col_span > 1:
            label += f"(×{cell.col_span})"
        cv2.putText(warped_view, label, (cx + 6, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("warped_grid", warped_view)


def visualize_detection(machine_image: np.ndarray,
                        items: list,
                        window_points: np.ndarray,
                        grid: GridResult | None) -> None:
    """Show detected OBBs on the original image then the rectified grid view."""
    vis = machine_image.copy()
    draw_obbs(vis, [item["obb"] for item in items], color=(0, 255, 0), thickness=2)
    for item in items:
        c = center_of_obb(item["obb"])
        cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)
    cv2.imshow("detection", vis)

    if grid is not None:
        visualize_grid(machine_image, window_points, grid)

    cv2.waitKey(-1)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def build_markdown_table(grid: GridResult) -> str:
    """
    Render the classified grid as a markdown table.

    Primary cells: <product_name> (<score:.2f>), with ×N suffix when merged.
    Continuation slots of merged cells: ←.  Empty slots: blank.
    """
    col_header = " | ".join(f"C{c}" for c in range(grid.n_cols))
    lines = [
        f"| Row | {col_header} |",
        "|-----|" + "-------|" * grid.n_cols,
    ]

    for r in range(grid.n_rows):
        parts    = [str(r)]
        seen_idx = None
        for c in range(grid.n_cols):
            idx = grid.grid_2d[r][c]
            if idx is None:
                parts.append("")
            elif idx == seen_idx:
                parts.append("←")
            else:
                cell  = grid.cells[idx]
                name  = cell.product_name or "?"
                label = f"{name} ({cell.product_score:.2f})"
                if cell.col_span > 1:
                    label += f" ×{cell.col_span}"
                parts.append(label)
            seen_idx = idx
        lines.append("| " + " | ".join(parts) + " |")

    return "\n".join(lines)