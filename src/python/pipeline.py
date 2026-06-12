from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from shared import (
    MODEL_PATHES,
    MACHINE_CLASSES,
    MACHINE_DETECTOR_CONF,
    MACHINE_DETECTOR_IOU,
    MACHINE_DETECTOR_IMGSZ,
    MACHINE_CLASSIFIER_CONF,
    MACHINE_CLASSIFIER_IMGSZ,
    ITEM_DETECTOR_CONF,
    ITEM_DETECTOR_IOU,
    ITEM_DETECTOR_IMGSZ,
    ITEM_MERGE_IOU,
)
from window_segmentator import WindowSegmentator
from grid_helper import (
    GridResult,
    build_grid,
    crop_obb_rotated,
    build_markdown_table,
    merge_overlapping_items,
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
        return self.machine_detector.predict(image, verbose=False, conf=MACHINE_DETECTOR_CONF, iou=MACHINE_DETECTOR_IOU, imgsz=MACHINE_DETECTOR_IMGSZ)[0].boxes.xyxy

    def _get_machine_classification(self, machine_image):
        result = self.machine_classificator.predict(machine_image, verbose=False, conf=MACHINE_CLASSIFIER_CONF, imgsz=MACHINE_CLASSIFIER_IMGSZ)[0]
        probs  = result.probs.data
        print(probs)
        return torch.argmax(probs).item()

    def _extract_window_points(self, image):
        return self.window_segmentator.getPoly(image)

    def _detect_items(self, image):
        result = self.item_detector.predict(image, verbose=False, conf=ITEM_DETECTOR_CONF, iou=ITEM_DETECTOR_IOU, imgsz=ITEM_DETECTOR_IMGSZ)[0]
        if result.obb is not None:
            return result.obb.xywhr
        # plain bbox model: pad xywh with a zero rotation column so downstream
        # grid code (which expects xywhr) works unchanged
        boxes = result.boxes.xywh
        return torch.cat([boxes, torch.zeros((boxes.shape[0], 1), device=boxes.device)], dim=1)

    def _classify_items(self, detection: MachineDetection) -> None:
        for cell in detection.grid.cells:
            crop = crop_obb_rotated(detection.image, cell.obb)
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
            items_list = merge_overlapping_items(items_list, ITEM_MERGE_IOU)
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

