import json
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
        self.spread_path  = self.gallery_path.with_name(self.gallery_path.stem + "_spread.json")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = _Encoder(embedding_size).to(self.device)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device, weights_only=False)
            # Lightning checkpoint includes arcface_head/criterion — ignore with strict=False
            self._model.load_state_dict(state, strict=False)
        self._model.eval()

        self._names:  list[str]  | None = None
        self._matrix: np.ndarray | None = None   # (N_classes, embedding_size)
        self._spread: dict[str, float]  = {}     # class_name -> mean intra-class cosine distance
        # guards _names/_matrix/_spread so a background recompute_class() (e.g. run
        # from a worker thread) can't be observed mid-swap by lookup_topk()
        self._lock = threading.RLock()

        if self.gallery_path.exists():
            self._load_gallery()
        if self.spread_path.exists():
            with open(self.spread_path) as f:
                self._spread = json.load(f)

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

    def _save_spread(self) -> None:
        self.spread_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.spread_path, "w") as f:
            json.dump(self._spread, f)

    @staticmethod
    def _mean_intra_distance(embeddings: list[np.ndarray], mean: np.ndarray) -> float:
        """Mean cosine distance of each (L2-normalised) embedding to the
        class's mean embedding direction."""
        mean_n = mean / max(float(np.linalg.norm(mean)), 1e-12)
        return float(np.mean([1.0 - float(mean_n @ e) for e in embeddings]))

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        """All class names currently loaded in the gallery."""
        with self._lock:
            return list(self._names) if self._names else []

    def clear_gallery(self) -> None:
        """Delete the saved embedding gallery (.npy and spread cache) and
        reset in-memory state, e.g. when it's found to be stale relative to
        the gallery directory on disk."""
        if self.gallery_path.exists():
            self.gallery_path.unlink()
        if self.spread_path.exists():
            self.spread_path.unlink()
        with self._lock:
            self._names  = None
            self._matrix = None
            self._spread = {}

    @property
    def spread_class_names(self) -> set[str]:
        """Class names that have mean intra-class distance stats."""
        with self._lock:
            return set(self._spread.keys())

    @property
    def global_mean_intra_class_distance(self) -> float | None:
        """Mean, across all classes with spread stats, of each class's mean
        intra-class cosine distance. None if no stats are available yet."""
        with self._lock:
            if not self._spread:
                return None
            return float(np.mean(list(self._spread.values())))

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Compute the L2-normalised embedding for a product crop."""
        return self._embed(crop_bgr)

    def class_mean_distance(self, class_name: str, embedding: np.ndarray) -> float | None:
        """Cosine distance between `embedding` and `class_name`'s stored mean
        embedding, or None if the class isn't in the gallery yet."""
        with self._lock:
            if not self._names or class_name not in self._names:
                return None
            mean_vec = self._matrix[self._names.index(class_name)]
        return float(1.0 - mean_vec @ embedding)

    def build_cache(self, gallery_dir: str) -> None:
        """
        Build and save the embedding gallery.

        Parameters
        ----------
        gallery_dir : str
            ImageFolder-style root: <gallery_dir>/<class_name>/<image.*>,
            each image already a single-item crop (as saved by
            gallery_labeler.py).
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
                embeddings.append(self._embed(img))

            if embeddings:
                mean = np.mean(embeddings, axis=0)
                gallery_data[class_dir.name] = mean
                self._spread[class_dir.name] = self._mean_intra_distance(embeddings, mean)
                print(f"  [{len(embeddings):3d}] {class_dir.name}")

        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.gallery_path), gallery_data, allow_pickle=True)
        self._save_spread()
        self._load_gallery()
        print(f"Gallery saved → {self.gallery_path}  ({len(gallery_data)} classes)")

    def recompute_class(self,
                         class_name:   str,
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
            embeddings.append(self._embed(img))
            if progress_cb:
                progress_cb(class_name, i + 1, len(img_paths))

        if not embeddings:
            return False

        mean = np.mean(embeddings, axis=0).astype(np.float32)
        data: dict[str, np.ndarray] = {}
        if self.gallery_path.exists():
            data = np.load(self.gallery_path, allow_pickle=True).item()
        data[class_name] = mean
        self._spread[class_name] = self._mean_intra_distance(embeddings, mean)

        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.gallery_path), data, allow_pickle=True)
        self._save_spread()
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