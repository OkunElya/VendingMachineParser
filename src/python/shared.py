MODEL_PATHES = {
    "machine_detector": "./models/tuned/vending_machine_detect_yolov10n.pt",
    "machine_classificator": "./models/tuned/vending_machine_classification_yolo26n-cls.pt",
    "window_segmentator": "./models/tuned/window_segmentation_yolo26n-seg.pt",
    "item_detector": "./models/tuned/items_detect.yolo26n-obb.pt",
    "item_classificator": "",
}

MACHINE_CLASSES = {
    0: {
        "name": "New Polka Kofe",
        "max_rows": 6,
        "max_cols": 8,
    },
    1: {
        "name": "New White Chiniece",
        "max_rows": 6,
        "max_cols": 8,
    },
    2: {
        "name": "Uvenco Foodbox",
        "max_rows": 6,
        "max_cols": 8,
    },
    3: {
        "name": "Uvenco Foodbox Wide",
        "max_rows": 6,
        "max_cols": 12,
    },
    4: {
        "name": "Uvenco Foodbox Wide New Skin",
        "max_rows": 6,
        "max_cols": 12,
    },

}
