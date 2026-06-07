from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from shared import MODEL_PATHES, MACHINE_CLASSES
from window_segmentator import WindowSegmentator
from grid_helper import (
    GridResult,
    build_grid,
    order_corners,
    warp_image,
    build_markdown_table,
    visualize_detection,
)
from item_classification import ProductBank


@dataclass
class MachineDetection:
    machine_info:  dict
    machine_bbox:  Any             # cpu tensor, xyxy
    window_points: np.ndarray
    items:         list            # list of {"obb": tensor}
    image:         np.ndarray      # copy of the cropped machine image
    grid:          GridResult | None = None


class Pipeline:
    def __init__(self):
        self.machine_detector      = YOLO(MODEL_PATHES["machine_detector"])
        self.machine_classificator = YOLO(MODEL_PATHES["machine_classificator"])
        self.window_segmentator    = WindowSegmentator()
        self.item_detector         = YOLO(MODEL_PATHES["item_detector"])
        self.product_bank          = ProductBank()

    def _detect_machines(self, image):
        return self.machine_detector.predict(image, verbose=False, conf=0.25)[0].boxes.xyxy

    def _get_machine_classification(self, machine_image):
        result = self.machine_classificator.predict(machine_image, verbose=False, conf=0.25)[0]
        probs  = result.probs.data
        print(probs)
        return torch.argmax(probs).item()

    def _extract_window_points(self, image):
        return self.window_segmentator.getPoly(image)

    def _detect_items(self, image):
        return self.item_detector.predict(image, verbose=False, conf=0.35)[0].obb.xywhr

    def _classify_items(self, detection: MachineDetection) -> None:
        grid    = detection.grid
        ordered = order_corners(detection.window_points)
        warped  = warp_image(detection.image, ordered, grid.out_w, grid.out_h)
        for cell in grid.cells:
            y0   = max(0, int(cell.top_y))
            y1   = min(grid.out_h, int(cell.bottom_y))
            x0   = max(0, int(cell.left_x))
            x1   = min(grid.out_w, int(cell.right_x))
            crop = warped[y0:y1, x0:x1]
            if crop.size > 0:
                cell.product_name, cell.product_score = self.product_bank.lookup(crop)
            else:
                cell.product_name, cell.product_score = None, 0.0

    def detect(self, image) -> list[MachineDetection]:
        results: list[MachineDetection] = []

        for machine_bb in self._detect_machines(image):
            machine_bb_img   = image[int(machine_bb[1]):int(machine_bb[3]),
                                     int(machine_bb[0]):int(machine_bb[2]), ::].copy()
            machine_class_id = self._get_machine_classification(machine_bb_img)
            window_points    = self._extract_window_points(machine_bb_img)

            if window_points is None:
                print("failed to segment window")
                continue

            items_list = [{"obb": obb} for obb in self._detect_items(machine_bb_img).cpu()]
            grid       = build_grid(MACHINE_CLASSES[machine_class_id], window_points, items_list)

            detection = MachineDetection(
                machine_info=MACHINE_CLASSES[machine_class_id],
                machine_bbox=machine_bb.cpu(),
                window_points=window_points,
                items=items_list,
                image=machine_bb_img,
                grid=grid,
            )

            if grid is not None:
                self._classify_items(detection)
            results.append(detection)

        return results

    def visualize(self, detection: MachineDetection) -> None:
        """Display OBBs and warped grid for a single detection result."""
        visualize_detection(detection.image, detection.items,
                            detection.window_points, detection.grid)


if __name__ == "__main__":
    import cv2
    pipeline = Pipeline()
    results = pipeline.detect(cv2.imread("./workspace/kartinki-dlya-sajta_rosso-bar.webp"))
    for det in results:
        pipeline.visualize(det)
        print(build_markdown_table(det.grid))

