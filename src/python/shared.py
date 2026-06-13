import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

MODEL_PATHES = {
    "machine_detector": "./models/tuned/vending_machine_detect_yolov10n.pt",
    "machine_classificator": "./models/tuned/vending_machine_classification_yolo26n-cls.pt",
    "window_segmentator": "./models/tuned/window_segmentation_yolo26n-seg.pt",
    "item_detector": "./models/tuned/items_detect_yolo26s-obb.pt",
    "item_classificator": "./models/tuned/items_classification_convnext_tiny.fb_in22k_ft_in1k.pt",
}

ITEM_GALLERY_PATH           = "./models/tuned/items_classification.npy"
GALLERY_DIR                 = "./gallery"
ITEM_CLASSIFICATION_BACKBONE = "hf_hub:timm/convnext_tiny.fb_in22k_ft_in1k"
ITEM_EMBEDDING_SIZE         = 512
ITEM_INPUT_SIZE             = 224   # encoder was fine-tuned on images padded-to-square then resized to this

CONFIG_PATH = "./config.json"


@dataclass
class Config:
    """Detection / merge thresholds and YOLO input sizes, persisted to
    `config.json` so they can be tuned (e.g. via scripts/tune_item_detector.py)
    without editing source. Defaults below match the previously hardcoded
    values; where pipeline.py didn't pass an explicit value, the ultralytics
    default is made explicit instead."""
    machine_detector_conf:    float = 0.3
    machine_detector_iou:     float = 0.5
    machine_classifier_conf:  float = 0.25
    item_detector_conf:       float = 0.25
    item_detector_iou:        float = 0.5
    item_merge_iou:           float = 0.3
    item_merge_containment:   float = 0.9   # drop an item if this fraction of its area is covered by a larger item

    # Input image size passed to each YOLO model's predict() call. Lower
    # values cut inference time at the cost of accuracy (smaller objects
    # become harder to detect/classify).
    machine_detector_imgsz:   int = 736
    machine_classifier_imgsz: int = 224
    window_segmentator_imgsz: int = 736
    item_detector_imgsz:      int = 736

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "Config":
        if not Path(path).is_file():
            return cls()
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


CONFIG = Config.load()

# Detection / classification thresholds (shared between pipeline.py and scripts).
MACHINE_DETECTOR_CONF   = CONFIG.machine_detector_conf
MACHINE_DETECTOR_IOU    = CONFIG.machine_detector_iou
MACHINE_CLASSIFIER_CONF = CONFIG.machine_classifier_conf
ITEM_DETECTOR_CONF      = CONFIG.item_detector_conf
ITEM_DETECTOR_IOU       = CONFIG.item_detector_iou
ITEM_MERGE_IOU          = CONFIG.item_merge_iou
ITEM_MERGE_CONTAINMENT  = CONFIG.item_merge_containment

MACHINE_DETECTOR_IMGSZ   = CONFIG.machine_detector_imgsz
MACHINE_CLASSIFIER_IMGSZ = CONFIG.machine_classifier_imgsz
WINDOW_SEGMENTATOR_IMGSZ = CONFIG.window_segmentator_imgsz
ITEM_DETECTOR_IMGSZ      = CONFIG.item_detector_imgsz

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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
