import os
import random
import shutil

DATASET_ROOT = "datasets/vending_machine_classification"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    random.seed(SEED)
    classes = sorted(
        d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))
    )

    for cls in classes:
        cls_train_dir = os.path.join(TRAIN_DIR, cls)
        images = sorted(
            f for f in os.listdir(cls_train_dir) if f.lower().endswith(IMG_EXTS)
        )
        random.shuffle(images)

        n = len(images)
        n_val = max(1, round(n * VAL_RATIO))
        n_test = max(1, round(n * TEST_RATIO))

        val_imgs = images[:n_val]
        test_imgs = images[n_val:n_val + n_test]
        train_imgs = images[n_val + n_test:]

        val_cls_dir = os.path.join(VAL_DIR, cls)
        test_cls_dir = os.path.join(TEST_DIR, cls)
        os.makedirs(val_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        for img in val_imgs:
            shutil.move(os.path.join(cls_train_dir, img), os.path.join(val_cls_dir, img))
        for img in test_imgs:
            shutil.move(os.path.join(cls_train_dir, img), os.path.join(test_cls_dir, img))

        print(f"{cls}: total={n} -> train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    # Stale caches reference the old train/val layout; remove so ultralytics rebuilds them.
    for cache in ("train.cache", "val.cache", "test.cache"):
        cache_path = os.path.join(DATASET_ROOT, cache)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"removed stale {cache_path}")


if __name__ == "__main__":
    main()
