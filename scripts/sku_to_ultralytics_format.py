import os
import pandas as pd

# Define paths matching your structure
dataset_root = "./datasets/SKU110K_fixed"
annotations_dir = os.path.join(dataset_root, "annotations")
images_dir = os.path.join(dataset_root, "images")

# Map files to split directories expected by Ultralytics
splits = {
    'train': 'annotations_train.csv',
    'val': 'annotations_val.csv',
    'test': 'annotations_test.csv'
}

def convert_to_obb_labels():
    for split_name, csv_file in splits.items():
        csv_path = os.path.join(annotations_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"Skipping {csv_file}, file not found.")
            continue
            
        print(f"Processing {split_name} split...")
        
        # Target directory for label txt files
        labels_output_dir = os.path.join(dataset_root, "labels", split_name)
        os.makedirs(labels_output_dir, exist_ok=True)
        
        # Load CSV (SKU110K headers: image_name, x1, y1, x2, y2, class, width, height)
        df = pd.read_csv(csv_path, names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class_name', 'img_w', 'img_h'])
        
        # Group annotations by image
        grouped = df.groupby('image_name')
        
        for img_name, group in grouped:
            txt_filename = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(labels_output_dir, txt_filename)
            
            with open(txt_path, 'w') as f:
                for _, row in group.iterrows():
                    # SKU110K usually treats everything as one class ("object") -> class index 0
                    class_idx = 0 
                    
                    w = float(row['img_w'])
                    h = float(row['img_h'])
                    
                    # Convert axis-aligned box to 4 corners
                    x1, y1 = float(row['x1']), float(row['y1'])
                    x2, y2 = float(row['x2']), float(row['y2'])
                    
                    # 4 Corners (Clockwise: Top-Left, Top-Right, Bottom-Right, Bottom-Left)
                    # Normalized by dividing by image dimensions
                    pts = [
                        x1 / w, y1 / h,  # x1, y1
                        x2 / w, y1 / h,  # x2, y1
                        x2 / w, y2 / h,  # x2, y2
                        x1 / w, y2 / h   # x1, y2
                    ]
                    
                    # Format: class x1 y1 x2 y2 x3 y3 x4 y4
                    line = f"{class_idx} " + " ".join(f"{p:.6f}" for p in pts) + "\n"
                    f.write(line)
                    
        print(f"Finished generating labels for {split_name}.")

import os
import shutil

# Define paths matching your structure
dataset_root = "./datasets/SKU110K_fixed"
images_src_dir = os.path.join(dataset_root, "images")
labels_root_dir = os.path.join(dataset_root, "labels")

# The splits we want to organize
splits = ['train', 'val', 'test']

def organize_images_by_split():
    print("Starting image reorganization...")
    
    # Track metrics for a quick summary at the end
    moved_counts = {split: 0 for split in splits}
    missing_counts = {split: 0 for split in splits}

    for split in splits:
        split_labels_dir = os.path.join(labels_root_dir, split)
        split_images_dir = os.path.join(images_src_dir, split)
        
        # Ensure target image directories (e.g., images/train) exist
        os.makedirs(split_images_dir, exist_ok=True)
        
        if not os.path.exists(split_labels_dir):
            print(f"Skipping split '{split}': Label directory not found.")
            continue
            
        print(f"Organizing images for '{split}' split...")
        
        # Iterate through all generated .txt files in the labels subfolder
        for label_file in os.listdir(split_labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            # Deduce image name from text file (SKU110K uses .jpg)
            image_name = os.path.splitext(label_file)[0] + ".jpg"
            
            # Source path (flat root of images folder) and destination path
            src_image_path = os.path.join(images_src_dir, image_name)
            dst_image_path = os.path.join(split_images_dir, image_name)
            
            if os.path.exists(src_image_path):
                # Move the image safely into its split folder
                shutil.move(src_image_path, dst_image_path)
                moved_counts[split] += 1
            elif os.path.exists(dst_image_path):
                # Handle case where script might have been partially run before
                moved_counts[split] += 1
            else:
                missing_counts[split] += 1

    # Print a status breakdown
    print("\nReorganization Complete Summary:")
    for split in splits:
        print(f"  - {split}: Moved {moved_counts[split]} images successfully. (Missing source files: {missing_counts[split]})")

if __name__ == "__main__":
    convert_to_obb_labels()    
    organize_images_by_split()
    