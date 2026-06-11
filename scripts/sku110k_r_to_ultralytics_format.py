import json
import os

# Define paths matching your structure
dataset_root = "./datasets/SKU110K_fixed"
images_dir = os.path.join(dataset_root, "images")
labels_dir = os.path.join(dataset_root, "labels")

# Map splits to the SKU110K-R COCO-style annotation files at the project root
splits = {
    'train': './sku110k-r_train.json',
    'val': './sku110k-r_val.json',
    'test': './sku110k-r_test.json',
}


def convert_to_obb_labels():
    for split_name, json_path in splits.items():
        if not os.path.exists(json_path):
            print(f"Skipping {json_path}, file not found.")
            continue

        print(f"Processing {split_name} split...")

        split_images_dir = os.path.join(images_dir, split_name)
        split_labels_dir = os.path.join(labels_dir, split_name)
        os.makedirs(split_labels_dir, exist_ok=True)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # image_id -> (file_name, width, height)
        images_by_id = {img['id']: img for img in data['images']}

        # Group annotations by image_id
        anns_by_image = {}
        for ann in data['annotations']:
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

        written, missing = 0, 0
        for image_id, anns in anns_by_image.items():
            img = images_by_id.get(image_id)
            if img is None:
                continue

            file_name = img['file_name']
            if not os.path.exists(os.path.join(split_images_dir, file_name)):
                missing += 1
                continue

            w, h = float(img['width']), float(img['height'])
            txt_filename = os.path.splitext(file_name)[0] + ".txt"
            txt_path = os.path.join(split_labels_dir, txt_filename)

            with open(txt_path, 'w') as out:
                for ann in anns:
                    # Single class dataset -> class index 0 ("object")
                    class_idx = 0

                    # 4 corners of the rotated quad, normalized by image dimensions
                    pts = ann['segmentation'][0]
                    norm_pts = [pts[i] / w if i % 2 == 0 else pts[i] / h
                                 for i in range(8)]

                    line = f"{class_idx} " + " ".join(f"{p:.6f}" for p in norm_pts) + "\n"
                    out.write(line)
            written += 1

        print(f"Finished {split_name}: wrote {written} label files, "
              f"skipped {missing} images not found on disk.")


if __name__ == "__main__":
    convert_to_obb_labels()
