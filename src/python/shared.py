MODEL_PATHES = {
    "machine_detector": "./models/tuned/vending_machine_detect_yolov10n.pt",
    "machine_classificator": "./models/tuned/vending_machine_classification_yolo26n-cls.pt",
    "window_segmentator": "./models/tuned/window_segmentation_yolo26n-seg.pt",
    "item_detector": "./models/tuned/items_detect.yolo26s-obb.pt",
    "item_classificator": "./models/tuned/items_classification_convnext_tiny.fb_in22k_ft_in1k.pt",
}

ITEM_GALLERY_PATH           = "./models/tuned/items_classification.npy"
GALLERY_DIR                 = "./gallery"
ITEM_CLASSIFICATION_BACKBONE = "hf_hub:timm/convnext_tiny.fb_in22k_ft_in1k"
ITEM_EMBEDDING_SIZE         = 512
ITEM_INPUT_SIZE             = 224   # encoder was fine-tuned on images padded-to-square then resized to this

# Detection / classification thresholds (shared between pipeline.py and scripts).
# Where pipeline.py didn't pass an explicit value, the ultralytics default is
# made explicit here instead.
MACHINE_DETECTOR_CONF   = 0.3
MACHINE_DETECTOR_IOU    = 0.5
MACHINE_CLASSIFIER_CONF = 0.25
ITEM_DETECTOR_CONF      = 0.2
ITEM_DETECTOR_IOU       = 0.5
ITEM_MERGE_IOU          = 0.3

# Input image size passed to each YOLO model's predict() call. Lower values
# cut inference time at the cost of accuracy (smaller objects become harder
# to detect/classify).
MACHINE_DETECTOR_IMGSZ   = 736
MACHINE_CLASSIFIER_IMGSZ = 224
WINDOW_SEGMENTATOR_IMGSZ = 736
ITEM_DETECTOR_IMGSZ      = 736

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
