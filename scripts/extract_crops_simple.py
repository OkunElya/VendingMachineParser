"""
Extract cropped images from bounding boxes in YOLO detection dataset.
Simple version for your folder structure.
"""

import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_bbox_crops(
    dataset_path="./detection",
    split="train",
    output_dir="./crops",
    class_names=None,
    padding=0.1
):
    """
    Extract bounding box crops from YOLO dataset.
    
    Args:
        dataset_path: Path to detection dataset root (e.g., ./detection)
        split: 'train' or 'val' or 'test'
        output_dir: Where to save cropped images
        class_names: List of class names
        padding: Extra padding around bbox (0.1 = 10%)
    """
    
    # Construct paths for your structure
    base = Path(dataset_path)
    img_dir = base / split / "images"
    label_dir = base / split / "labels"
    output_path = Path(output_dir)
    
    # Validate directories
    if not img_dir.exists():
        print(f"❌ Images directory not found: {img_dir}")
        print(f"   Expected: {img_dir.absolute()}")
        return
    
    if not label_dir.exists():
        print(f"❌ Labels directory not found: {label_dir}")
        return
    
    # Default class names if not provided
    if class_names is None:
        class_names = {}
    
    # Convert list to dict
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(img_dir.glob("*"))
    image_files = [f for f in image_files 
                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    print(f"✅ Found {len(image_files)} images")
    print(f"📁 Images: {img_dir}")
    print(f"📁 Labels: {label_dir}")
    
    crop_count = 0
    
    for img_file in tqdm(image_files, desc="Extracting crops"):
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Find corresponding label file
        label_file = label_dir / (img_file.stem + ".txt")
        
        if not label_file.exists():
            continue
        
        # Read annotations
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Extract each bounding box
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])
            except ValueError:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            x_center *= w
            y_center *= h
            bbox_w *= w
            bbox_h *= h
            
            # Add padding
            pad_w = bbox_w * padding
            pad_h = bbox_h * padding
            
            x_min = max(0, int(x_center - bbox_w/2 - pad_w))
            y_min = max(0, int(y_center - bbox_h/2 - pad_h))
            x_max = min(w, int(x_center + bbox_w/2 + pad_w))
            y_max = min(h, int(y_center + bbox_h/2 + pad_h))
            
            # Extract crop
            crop = img[y_min:y_max, x_min:x_max]
            
            if crop.size == 0:
                continue
            
            # Create class folder
            class_name = class_names.get(class_id, f"class_{class_id}")
            class_dir = output_path / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Save crop
            crop_filename = f"{img_file.stem}_{idx}.jpg"
            crop_path = class_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)
            
            crop_count += 1
    
    print(f"\n✅ Extracted {crop_count} crops")
    print(f"📁 Saved to: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract bounding box crops from YOLO dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./detection",
        help="Path to detection dataset root"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./crops",
        help="Output directory for crops"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Padding around bbox (0.1 = 10%)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Class names (e.g., --classes VendingMachine Window Body)"
    )
    
    args = parser.parse_args()
    
    extract_bbox_crops(
        dataset_path=args.dataset,
        split=args.split,
        output_dir=args.output,
        class_names=args.classes,
        padding=args.padding
    )


if __name__ == "__main__":
    main()
