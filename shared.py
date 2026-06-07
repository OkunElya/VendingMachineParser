MODEL_PATHES = {
    "machine_detector": "./models/tuned/vending_machine_detect_yolov10n.pt",
    "machine_classificator": "./models/tuned/vending_machine_classification_yolo26n-cls.pt",
    "window_segmentator": "./models/tuned/window_segmentation_yolo26n-seg.pt",
    "item_detector": "./models/tuned/items_detect.yolo26n-obb.pt",
    "item_classificator": "",
}

MACHINE_CLASSES = {
    0:{
        "name":"UniCum FoodBox",
        "max_rows":6,
        "max_cols":8,
    },
    1:{
        "name":"Chiniece White",
        "max_rows":8,
        "max_cols":8,
    }
}