import math
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms.v2 as transforms
from PIL import Image
from pathlib import Path

_WEIGHTS_PATH   = "./models/tuned/items_classification_convnext_tiny.fb_in22k_ft_in1k.pt"
_GALLERY_PATH   = "./models/tuned/items_classification.npy"
_BACKBONE_ID    = "hf_hub:timm/convnext_tiny.fb_in22k_ft_in1k"
_EMBEDDING_SIZE = 512
_INPUT_SIZE     = 224   # encoder was fine-tuned on images padded-to-square then resized to this
_IMG_EXTS       = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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
    def __init__(self, embedding_size: int = _EMBEDDING_SIZE):
        super().__init__()
        self.backbone = timm.create_model(_BACKBONE_ID, pretrained=False)
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
                 model_path:     str               = _WEIGHTS_PATH,
                 gallery_path:   str               = _GALLERY_PATH,
                 embedding_size: int               = _EMBEDDING_SIZE,
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

        if self.gallery_path.exists():
            self._load_gallery()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_gallery(self) -> None:
        data = np.load(self.gallery_path, allow_pickle=True).item()
        self._names  = list(data.keys())
        self._matrix = np.stack([data[k] for k in self._names]).astype(np.float32)

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
        square = cv2.resize(square, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb    = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self._model(tensor).cpu().numpy().flatten()

    def _crop_largest_obb(self, img: np.ndarray, item_detector) -> np.ndarray:
        """Run detector and return an axis-aligned crop of the largest OBB."""
        from src.python.grid_helper import obb_xywhr_to_corners
        try:
            obbs = item_detector(img)
            if obbs is not None and len(obbs):
                areas = [float(o[2]) * float(o[3]) for o in obbs]
                best  = obbs[int(np.argmax(areas))]
                pts   = obb_xywhr_to_corners(best)
                x0 = max(0, int(pts[:, 0].min()))
                y0 = max(0, int(pts[:, 1].min()))
                x1 = min(img.shape[1], int(pts[:, 0].max()))
                y1 = min(img.shape[0], int(pts[:, 1].max()))
                crop = img[y0:y1, x0:x1]
                if crop.size > 0:
                    return crop
        except Exception:
            pass
        return img

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
                if img_path.suffix.lower() not in _IMG_EXTS:
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

    def lookup(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        """
        Classify a product crop (any dimensions).

        Returns (class_name, cosine_similarity) or (None, 0.0) if no gallery.
        """
        if self._matrix is None or not self._names:
            return None, 0.0
        emb    = self._embed(crop_bgr)
        scores = self._matrix @ emb        # dot product of L2-normalised vecs = cosine sim
        best   = int(np.argmax(scores))
        return self._names[best], float(scores[best])