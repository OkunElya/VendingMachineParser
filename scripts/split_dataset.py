import os
import shutil
import random
from PIL import Image

def pad_to_square(image_path, dest_path, size=(224, 224)):
    """Pads image to square preserving aspect ratio and resizes it."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            max_wh = max(w, h)
            
            # Create a solid black canvas background
            squared_img = Image.new('RGB', (max_wh, max_wh), (0, 0, 0))
            
            # Center the original rectangle image on the canvas
            offset = ((max_wh - w) // 2, (max_wh - h) // 2)
            squared_img.paste(img, offset)
            
            # Resize cleanly to uniform model target dims
            squared_img = squared_img.resize(size, Image.Resampling.BILINEAR)
            squared_img.save(dest_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def build_arcface_dataset(src_root, dest_root, split_ratio=0.8):
    categories = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    
    for cat in categories:
        cat_path = os.path.join(src_root, cat)
        web_root = os.path.join(cat_path, 'web')
        shelf_root = os.path.join(cat_path, 'shelf')
        
        if not os.path.exists(shelf_root):
            continue
            
        # Process Class Folders within Shelf
        classes = [c for c in os.listdir(shelf_root) if os.path.isdir(os.path.join(shelf_root, c))]
        
        for cls in classes:
            # Setup structured destination paths
            train_cls_dir = os.path.join(dest_root, 'train', cls)
            val_cls_dir = os.path.join(dest_root, 'val', cls)
            gallery_cls_dir = os.path.join(dest_root, 'gallery', cls)
            
            os.makedirs(train_cls_dir, exist_ok=True)
            os.makedirs(val_cls_dir, exist_ok=True)
            os.makedirs(gallery_cls_dir, exist_ok=True)
            
            # 1. Process real-world Shelf images (Train/Val Split)
            cls_shelf_path = os.path.join(shelf_root, cls)
            shelf_images = [img for img in os.listdir(cls_shelf_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            random.seed(42)
            random.shuffle(shelf_images)
            split_idx = int(len(shelf_images) * split_ratio)
            
            train_imgs = shelf_images[:split_idx]
            val_imgs = shelf_images[split_idx:]
            
            for img in train_imgs:
                pad_to_square(os.path.join(cls_shelf_path, img), os.path.join(train_cls_dir, img))
            for img in val_imgs:
                pad_to_square(os.path.join(cls_shelf_path, img), os.path.join(val_cls_dir, img))
                
            # 2. Process pristine Web images (Gallery Base)
            cls_web_path = os.path.join(web_root, cls)
            if os.path.exists(cls_web_path):
                web_images = [img for img in os.listdir(cls_web_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img in web_images:
                    pad_to_square(os.path.join(cls_web_path, img), os.path.join(gallery_cls_dir, img))

if __name__ == "__main__":
    # Adjust source directory path to point where beverage/, snacks/ folders sit
    SOURCE_DIRECTORY = "./datasets/Retail-YU"
    OUTPUT_DIRECTORY = "./datasets/Retail-YU_reformed"
    print("Beginning dataset parsing and square padding transforms...")
    build_arcface_dataset(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)
    print("Dataset distribution split successfully compiled.")