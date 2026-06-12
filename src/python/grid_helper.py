from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import MachineDetection

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


def warp_image_inv(img, corners, dst_w, dst_h):
    """Inverse of warp_image: warp a rectified `img` back into the coordinate
    space where `corners` describes its quadrilateral, producing a
    (dst_w, dst_h) image."""
    h, w = img.shape[:2]
    src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, corners.astype("float32"))
    return cv2.warpPerspective(img, M, (dst_w, dst_h))


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


# ---------------------------------------------------------------------------
# OBB cropping helpers
# ---------------------------------------------------------------------------

def _edge_mean_color(crop: np.ndarray):
    """Mean color along `crop`'s outer border, used as a neutral padding
    fill so padding blends into the surroundings instead of looking like a
    stark black edge."""
    edges = np.concatenate([
        crop[0].reshape(-1, *crop.shape[2:]),
        crop[-1].reshape(-1, *crop.shape[2:]),
        crop[:, 0].reshape(-1, *crop.shape[2:]),
        crop[:, -1].reshape(-1, *crop.shape[2:]),
    ], axis=0)
    return tuple(float(v) for v in np.atleast_1d(edges.mean(axis=0)))


def crop_obb_rotated(image: np.ndarray, obb, pad_square: bool = True) -> np.ndarray:
    """Crop the region covered by an OBB (xywhr, r in radians), rotating the
    image about the OBB's center so its w/h axes become axis-aligned (i.e.
    de-skewing it). The box is additionally canonicalised to landscape
    (w >= h) via an extra 90-degree rotation when needed, so padding always
    lands on the same pair of sides regardless of which axis the detector
    assigned as w vs h. Areas falling outside the source image are
    zero-filled. When `pad_square` is set, the result is further padded to a
    square (centred) using the crop's own edge-mean color.
    """
    x, y, w, h, r = float(obb[0]), float(obb[1]), float(obb[2]), float(obb[3]), float(obb[4])
    img_h, img_w = image.shape[:2]

    angle_deg = np.degrees(r)
    if h > w:
        angle_deg += 90.0
        w, h = h, w

    M       = cv2.getRotationMatrix2D((x, y), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (img_w, img_h), flags=cv2.INTER_LINEAR)

    w_i, h_i = max(1, int(round(w))), max(1, int(round(h)))
    x0, y0   = int(round(x - w_i / 2)), int(round(y - h_i / 2))

    crop = np.zeros((h_i, w_i) + image.shape[2:], dtype=image.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img_w, x0 + w_i), min(img_h, y0 + h_i)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        crop[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = rotated[sy0:sy1, sx0:sx1]

    if pad_square:
        side  = max(crop.shape[0], crop.shape[1])
        pad_t = (side - crop.shape[0]) // 2
        pad_b = side - crop.shape[0] - pad_t
        pad_l = (side - crop.shape[1]) // 2
        pad_r = side - crop.shape[1] - pad_l
        if pad_t or pad_b or pad_l or pad_r:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                      cv2.BORDER_CONSTANT, value=_edge_mean_color(crop))

    return crop


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
# OBB merging helpers
# ---------------------------------------------------------------------------

def obb_iou(obb1, obb2) -> float:
    """Exact IoU between two OBBs (xywhr) via convex-polygon intersection."""
    c1, c2 = obb_xywhr_to_corners(obb1), obb_xywhr_to_corners(obb2)
    area1, area2 = cv2.contourArea(c1), cv2.contourArea(c2)
    if area1 <= 0 or area2 <= 0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(c1, c2)
    if inter_area <= 0:
        return 0.0
    return float(inter_area / (area1 + area2 - inter_area))


def merge_obbs(obbs) -> np.ndarray:
    """Merge several OBBs (xywhr) into the single rotated rect (xywhr) that
    tightly encloses all of their corners."""
    corners = np.concatenate([obb_xywhr_to_corners(o) for o in obbs], axis=0)
    (cx, cy), (w, h), angle = cv2.minAreaRect(corners)
    return np.array([cx, cy, w, h, np.radians(angle)], dtype=np.float32)


def merge_overlapping_items(items: list, iou_threshold: float = 0.5) -> list:
    """Merge item OBBs whose pairwise IoU exceeds `iou_threshold` into a single
    bounding OBB. Items are transitively grouped, so chains of overlapping
    boxes collapse into one merged box per cluster."""
    n = len(items)
    if n <= 1:
        return items

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if obb_iou(items[i]["obb"], items[j]["obb"]) > iou_threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(items[idxs[0]])
        else:
            merged.append({"obb": merge_obbs([items[i]["obb"] for i in idxs])})
    return merged


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

def _align_row_borders(borders: list[float], ref: list[float]) -> list[int] | None:
    """
    Find the order-preserving injection from `borders` into indices of `ref`
    that minimises the total absolute horizontal distance between matched
    values (DP over monotonic matchings of two sorted sequences).

    Returns the matched ref-index for each border, or None when `borders`
    is longer than `ref` (no valid injection exists).
    """
    m, n = len(borders), len(ref)
    if m > n:
        return None

    INF  = float("inf")
    dp   = [[INF] * n for _ in range(m)]
    back = [[-1]  * n for _ in range(m)]

    for j in range(n):
        dp[0][j] = abs(borders[0] - ref[j])

    for i in range(1, m):
        best_j, best_cost = -1, INF
        for j in range(n):
            if j - 1 >= 0 and dp[i - 1][j - 1] < best_cost:
                best_cost, best_j = dp[i - 1][j - 1], j - 1
            if best_j != -1:
                dp[i][j]   = best_cost + abs(borders[i] - ref[j])
                back[i][j] = best_j

    j = min(range(n), key=lambda j: dp[m - 1][j])
    mapping = [0] * m
    for i in range(m - 1, -1, -1):
        mapping[i] = j
        j = back[i][j]
    return mapping


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
    threshold = 0.05 * y_span if y_span > 1.0 else 1.0

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
    
    center_point_widths=[]
    for group in row_groups:
        row_points = sorted(list([warped_centers[idx] for idx in group]), key = lambda x:x[0])
        for idx in range(len(row_points)-1):
            center_point_widths.append(row_points[idx+1]-row_points[idx])
            
            
    max_cols    = machine_info.get("max_cols", 10)
    bottom_25_n = max(1, N // 4)
    bottom_10_n = max(1, N // 10)
    
    # unit_width  = float(np.sort(warped_widths)[:bottom_25_n].mean())
    unit_width  = float(np.sort(center_point_widths)[bottom_10_n:bottom_25_n].mean())
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
            gap = left_xs[i] - last_right
            if gap > 0.5 * unit_width:
                # one or more empty cells sit between the previous item and this one
                borders.append(float(last_right))
                col += 1
                
            border_x =(left_xs[i]+last_right)/2
            if gap > 0.5 * unit_width:
                #after gap use item leftmost point as new border pos
                border_x=left_xs[i]
                
            borders.append(float(border_x))
            col_of[i] = col
            col += 1
            last_right = right_xs[i]
        borders.append(float(right_xs[row_sorted[-1]]))
        
        if col >   max_cols:# bad item alignment , ghost item at the beggining
            offset = col-max_cols 
            borders = borders[offset:]
            col-=offset
            for i in row_sorted:
                col_of[i]-=offset
        
        row_col_borders.append(borders)
        n_cols_per_row.append(col)

    n_cols = min(max_cols, max(n_cols_per_row))

    # --- Row boundary Y values -----------------------------------------------
    row_boundaries: list[float] = []
    for grp in row_groups:
        row_boundaries.append(float(top_ys[grp].min()))
    row_boundaries.append(float(bottom_ys[row_groups[-1]].max()))


    reference_rows = list(filter(lambda row : len(row) == max([len(row) for row in row_col_borders]),row_col_borders))
    min_val = min([row[0] for row in reference_rows])
    max_val = max([row[-1] for row in reference_rows])
    reference_rows.sort(key=lambda row:abs(min_val - row[0]) + abs(max_val - row[-1]))
    
    ref_row = reference_rows[0] # widest possible row

    for r, grp in enumerate(row_groups):
        row_sorted = sorted(grp, key=lambda i: left_xs[i])
        borders    = row_col_borders[r]
        mapping    = _align_row_borders(borders, ref_row)
        if mapping is None:
            continue

        for k,i in enumerate(row_sorted):
            item_row_idx=col_of[i]
            col_of[i]= mapping[item_row_idx]
            col_spans[i]= max(1, mapping[item_row_idx+1]-mapping[item_row_idx])


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

def _draw_grid_overlay(canvas: np.ndarray, grid: GridResult) -> np.ndarray:
    """Draw row/column grid lines and cell labels onto `canvas` in-place."""
    for y in grid.row_boundaries:
        cv2.line(canvas, (0, int(y)), (grid.out_w, int(y)),
                 (0, 255, 255), 1, cv2.LINE_AA)

    for r, borders in enumerate(grid.row_col_borders):
        y_top = int(grid.row_boundaries[r])
        y_bot = int(grid.row_boundaries[r + 1])
        for x in borders:
            cv2.line(canvas, (int(x), y_top), (int(x), y_bot),
                     (0, 200, 255), 1, cv2.LINE_AA)

    for cell in grid.cells:
        cx, cy = int(cell.center_warped[0]), int(cell.center_warped[1])
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), -1)
        label = f"{cell.row},{cell.col}"
        if cell.col_span > 1:
            label += f"(×{cell.col_span})"
        cv2.putText(canvas, label, (cx + 6, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    return canvas


def render_grid(machine_image: np.ndarray,
                window_points: np.ndarray,
                grid: GridResult) -> np.ndarray:
    """Draw row/column grid lines and cell labels onto the rectified window image."""
    ordered     = order_corners(window_points)
    warped_view = warp_image(machine_image, ordered, grid.out_w, grid.out_h)
    return _draw_grid_overlay(warped_view, grid)


def render_overlay(detection: MachineDetection) -> np.ndarray:
    """Render the grid, item OBBs (green), and window outline directly on the
    (unwarped) machine image, for display in the frontend.

    The grid is drawn on a blank rectified canvas and projected back onto the
    machine image with the inverse of the perspective transform used by
    `warp_image`/`render_grid`.
    """
    overlay = detection.image.copy()
    h, w    = overlay.shape[:2]
    ordered = order_corners(detection.window_points)

    if detection.grid is not None:
        grid       = detection.grid
        grid_layer = np.zeros((grid.out_h, grid.out_w, 3), dtype=np.uint8)
        _draw_grid_overlay(grid_layer, grid)

        unwarped = warp_image_inv(grid_layer, ordered, w, h)
        mask     = unwarped.any(axis=2)
        overlay[mask] = unwarped[mask]

    draw_obbs(overlay, [item["obb"] for item in detection.items], color=(0, 255, 0))

    window_poly = ordered.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [window_poly], isClosed=True,
                  color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    return overlay


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
        cv2.imshow("warped_grid", render_grid(machine_image, window_points, grid))

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