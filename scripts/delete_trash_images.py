import os
import glob
from PIL import Image, ImageFile

# Force PIL to raise errors for broken images instead of ignoring them
ImageFile.LOAD_TRUNCATED_IMAGES = False

dataset_images_path = "./datasets/SKU110K_fixed/images/**/*.jpg"
image_paths = glob.glob(dataset_images_path, recursive=True)

corrupted_images = []

print(f"Scanning {len(image_paths)} images for deep byte corruption...")

for img_path in image_paths:
    try:
        with Image.open(img_path) as img:
            # .verify() catches structural issues but misses internal byte data errors
            img.verify() 
            
        # Re-open and force full pixel loading to catch premature end of data segments
        with Image.open(img_path) as img:
            img.load() 
            
    except Exception as e:
        print(f"Corrupted image found: {img_path} | Error: {e}")
        corrupted_images.append(img_path)

print("\n--- Scan Results ---")
if corrupted_images:
    print(f"Found {len(corrupted_images)} corrupted images.")
    # Optional: Uncomment the lines below to auto-delete them and their labels
    # for img_to_delete in corrupted_images:
    #     os.remove(img_to_delete)
    #     label_to_delete = img_to_delete.replace("images", "labels").replace(".jpg", ".txt")
    #     if os.path.exists(label_to_delete):
    #         os.remove(label_to_delete)
    # print("Deleted corrupted images and matching labels.")
else:
    print("All images decoded perfectly via standard PIL loaders.")