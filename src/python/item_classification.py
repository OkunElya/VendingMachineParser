import math
import os
import threading
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms.v2 as transforms
from PIL import Image
from pathlib import Path

from shared import (
    MODEL_PATHES,
    ITEM_GALLERY_PATH,
    GALLERY_DIR,
    ITEM_CLASSIFICATION_BACKBONE,
    ITEM_EMBEDDING_SIZE,
    ITEM_INPUT_SIZE,
    IMAGE_EXTS,
)

_TRANSFORM = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ArcMarginProduct(nn.Module):
    """ArcFace margin layer — used during training only."""
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s  = s
        self.m  = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine  = F.linear(F.normalize(input), F.normalize(self.weight))
        sine    = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class _Encoder(nn.Module):
    """Inference-only backbone: ConvNeXt + BN + embedding projection."""
    def __init__(self, embedding_size: int = ITEM_EMBEDDING_SIZE):
        super().__init__()
        self.backbone = timm.create_model(ITEM_CLASSIFICATION_BACKBONE, pretrained=False)
        num_feat = self.backbone.num_features
        self.backbone.head.fc = nn.Identity()
        self.bn_layer     = nn.BatchNorm1d(num_feat)
        self.fc_embedding = nn.Linear(num_feat, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = self.bn_layer(f)
        return F.normalize(self.fc_embedding(f))


class ProductBank:
    """
    Embedding bank for product classification via cosine nearest-neighbour.

    build_cache(gallery_dir, item_detector)
        Walks gallery_dir/<class_name>/<image.*>, runs item_detector on each
        image, crops the largest detected OBB with zero-padding to square,
        extracts an embedding, averages per class, and saves to disk.

    lookup(crop_bgr)
        Zero-pads the crop to square, extracts an embedding, and returns the
        nearest class name plus cosine-similarity score.
    """

    def __init__(self,
                 model_path:     str               = MODEL_PATHES["item_classificator"],
                 gallery_path:   str               = ITEM_GALLERY_PATH,
                 embedding_size: int               = ITEM_EMBEDDING_SIZE,
                 device:         torch.device | None = None):
        self.gallery_path = Path(gallery_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = _Encoder(embedding_size).to(self.device)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device, weights_only=False)
            # Lightning checkpoint includes arcface_head/criterion — ignore with strict=False
            self._model.load_state_dict(state, strict=False)
        self._model.eval()

        self._names:  list[str]  | None = None
        self._matrix: np.ndarray | None = None   # (N_classes, embedding_size)
        # guards _names/_matrix so a background recompute_class() (e.g. run
        # from a worker thread) can't be observed mid-swap by lookup_topk()
        self._lock = threading.RLock()

        if self.gallery_path.exists():
            self._load_gallery()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_gallery(self) -> None:
        data  = np.load(self.gallery_path, allow_pickle=True).item()
        names = list(data.keys())
        matrix = np.stack([data[k] for k in names]).astype(np.float32)
        # gallery rows are means of L2-normalised embeddings, so they aren't
        # unit-length themselves; re-normalise so the dot product in lookup()
        # is a true cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)
        with self._lock:
            self._names  = names
            self._matrix = matrix

    def _pad_square(self, img_bgr: np.ndarray) -> np.ndarray:
        """Zero-pad a BGR image to a square (centred)."""
        h, w  = img_bgr.shape[:2]
        size  = max(h, w, 1)
        pad_t = (size - h) // 2
        pad_b = size - h - pad_t
        pad_l = (size - w) // 2
        pad_r = size - w - pad_l
        return cv2.copyMakeBorder(img_bgr, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def _embed(self, img_bgr: np.ndarray) -> np.ndarray:
        square = self._pad_square(img_bgr)
        square = cv2.resize(square, (ITEM_INPUT_SIZE, ITEM_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb    = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self._model(tensor).cpu().numpy().flatten()

    def _crop_largest_obb(self, img: np.ndarray, item_detector) -> np.ndarray:
        """Run detector and return a rotation-corrected, square-padded crop
        of the largest OBB."""
        from grid_helper import crop_obb_rotated
        try:
            obbs = item_detector(img)
            if obbs is not None and len(obbs):
                areas = [float(o[2]) * float(o[3]) for o in obbs]
                best  = obbs[int(np.argmax(areas))]
                crop  = crop_obb_rotated(img, best)
                if crop.size > 0:
                    return crop
        except Exception:
            pass
        return img

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        """All class names currently loaded in the gallery."""
        with self._lock:
            return list(self._names) if self._names else []

    def clear_gallery(self) -> None:
        """Delete the saved embedding gallery (.npy) and reset in-memory
        state, e.g. when it's found to be stale relative to the gallery
        directory on disk."""
        if self.gallery_path.exists():
            self.gallery_path.unlink()
        with self._lock:
            self._names  = None
            self._matrix = None

    def build_cache(self, gallery_dir: str, item_detector) -> None:
        """
        Build and save the embedding gallery.

        Parameters
        ----------
        gallery_dir : str
            ImageFolder-style root: <gallery_dir>/<class_name>/<image.*>
        item_detector : callable
            (image_bgr: np.ndarray) -> OBBs in xywhr format.
            Pass pipeline._detect_items.
        """
        root         = Path(gallery_dir)
        gallery_data: dict[str, np.ndarray] = {}

        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            embeddings = []
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                crop = self._crop_largest_obb(img, item_detector)
                embeddings.append(self._embed(crop))

            if embeddings:
                gallery_data[class_dir.name] = np.mean(embeddings, axis=0)
                print(f"  [{len(embeddings):3d}] {class_dir.name}")

        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.gallery_path), gallery_data, allow_pickle=True)
        self._load_gallery()
        print(f"Gallery saved → {self.gallery_path}  ({len(gallery_data)} classes)")

    def recompute_class(self,
                         class_name:   str,
                         item_detector,
                         gallery_dir:  str | None = None,
                         progress_cb=None) -> bool:
        """
        Recompute the embedding for a single class from its gallery images
        and write it back into the saved gallery, leaving every other
        class's embedding untouched (no full rebuild). Inserts a new row
        when `class_name` is not yet in the gallery.

        progress_cb(class_name, done, total), if given, is called after each
        image of the class is embedded.

        Returns True if the embedding was updated/inserted, False if the
        class directory has no usable images.
        """
        class_dir = Path(gallery_dir or GALLERY_DIR) / class_name
        img_paths = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS] \
            if class_dir.exists() else []

        embeddings = []
        for i, img_path in enumerate(img_paths):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            crop = self._crop_largest_obb(img, item_detector)
            embeddings.append(self._embed(crop))
            if progress_cb:
                progress_cb(class_name, i + 1, len(img_paths))

        if not embeddings:
            return False

        data: dict[str, np.ndarray] = {}
        if self.gallery_path.exists():
            data = np.load(self.gallery_path, allow_pickle=True).item()
        data[class_name] = np.mean(embeddings, axis=0).astype(np.float32)

        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.gallery_path), data, allow_pickle=True)
        self._load_gallery()
        return True

    def lookup(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        """
        Classify a product crop (any dimensions).

        Returns (class_name, cosine_similarity) or (None, 0.0) if no gallery.
        """
        with self._lock:
            matrix, names = self._matrix, self._names
        if matrix is None or not names:
            return None, 0.0
        emb    = self._embed(crop_bgr)
        scores = matrix @ emb        # dot product of L2-normalised vecs = cosine sim
        best   = int(np.argmax(scores))
        return names[best], float(scores[best])

    def lookup_topk(self, crop_bgr: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        Like lookup(), but returns up to `k` (class_name, cosine_similarity)
        matches ordered best-first. Returns [] if the gallery is empty.
        """
        with self._lock:
            matrix, names = self._matrix, self._names
        if matrix is None or not names:
            return []
        emb    = self._embed(crop_bgr)
        scores = matrix @ emb
        order  = np.argsort(scores)[::-1][:k]
        return [(names[i], float(scores[i])) for i in order]