"""Inference API for single-image aerial geometry estimation.

This is a self-contained, pip-installable version of the research script
``infer.py`` from the GeoReID codebase. Unlike the original script it does
not import any training code: the model architecture lives in
:mod:`aerialgeometry.model` and the normalization statistics are read from
the checkpoint itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .model import CLASS_TO_ANGLE, MultiTaskModel

__all__ = [
    "DEFAULT_TRANSFORM",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CACHE_PATH",
    "MODEL_URL",
    "Prediction",
    "Predictor",
    "default_device",
    "gather_images",
    "load_checkpoint",
    "resolve_model_path",
]

#: Preprocessing pipeline used at both training and inference time.
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

#: Directory where model weights are cached on first use.
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "aerialgeometry")
#: Default checkpoint location inside the cache directory.
DEFAULT_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "best.pth")

#: Optional URL from which weights are auto-downloaded on first use.
#: Set the ``AERIALGEOMETRY_MODEL_URL`` environment variable, e.g. to a
#: GitHub release asset or a Hugging Face file, and the checkpoint is fetched
#: into :data:`DEFAULT_CACHE_DIR` automatically.
MODEL_URL = os.environ.get("AERIALGEOMETRY_MODEL_URL", "")

#: Supported image extensions when scanning folders.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class Prediction:
    """Predicted geometry for a single image.

    Attributes:
        height: Predicted drone height above ground (meters on the REID dataset).
        distance: Predicted distance to the imaged subject (meters on the REID dataset).
        angle: Predicted viewing angle in degrees (one of 30, 60, 90).
        angle_probabilities: Softmax probabilities over the angle classes [30, 60, 90].
    """

    height: float
    distance: float
    angle: float
    angle_probabilities: np.ndarray


def default_device() -> torch.device:
    """Return the best available device (CUDA if present, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model_path: str, device: torch.device):
    """Load a checkpoint and return ``(state_dict, normalization_stats)``.

    Both full training checkpoints (containing a ``model_state`` key, as
    produced by the GeoReID ``train.py``) and plain state dicts are accepted.
    If the checkpoint does not carry ``normalization_stats``, identity
    normalization ``(0, 1, 0, 1)`` is used with a warning.
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        stats = ckpt.get("normalization_stats")
    else:
        state_dict = ckpt
        stats = None

    if stats is None:
        print(
            "WARNING: checkpoint has no 'normalization_stats'; "
            "falling back to identity normalization."
        )
        stats = (0.0, 1.0, 0.0, 1.0)
    return state_dict, tuple(float(v) for v in stats)


def _download_weights(url: str, dest: str) -> str:
    """Download model weights from ``url`` into ``dest`` with a progress bar."""
    import urllib.request

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "aerialgeometry/0.1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading weights") as bar:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)
    return dest


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolve the checkpoint location.

    Priority order:

    1. explicit ``model_path`` argument
    2. ``AERIALGEOMETRY_MODEL`` environment variable
    3. cached weights at :data:`DEFAULT_CACHE_PATH`
    4. automatic download from :data:`MODEL_URL` (``AERIALGEOMETRY_MODEL_URL``)
    """
    if model_path is not None:
        return model_path
    env_path = os.environ.get("AERIALGEOMETRY_MODEL")
    if env_path:
        return env_path
    if os.path.exists(DEFAULT_CACHE_PATH):
        return DEFAULT_CACHE_PATH
    if MODEL_URL:
        return _download_weights(MODEL_URL, DEFAULT_CACHE_PATH)
    raise FileNotFoundError(
        "No model weights found. Pass --model-path, set the "
        "AERIALGEOMETRY_MODEL environment variable, place a checkpoint at "
        f"{DEFAULT_CACHE_PATH}, or set AERIALGEOMETRY_MODEL_URL to have the "
        "weights auto-downloaded on first use."
    )


def gather_images(
    folder: Union[str, Path],
    recursive: bool = True,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> List[str]:
    """Collect image paths from a folder (sorted, deduplicated)."""
    files: List[str] = []
    folder = str(folder)
    for ext in extensions:
        pattern = os.path.join(folder, "**", f"*{ext}") if recursive else os.path.join(folder, f"*{ext}")
        files.extend(glob(pattern, recursive=recursive))
    return sorted(set(files))


class Predictor:
    """Loads a trained multi-task model and predicts geometry from images.

    Args:
        model_path: Path to the checkpoint. If ``None``, the location is
            resolved via :func:`resolve_model_path`.
        device: Torch device (``"cpu"``, ``"cuda:0"``, ...). Defaults to
            the best available device.
        transform: Preprocessing pipeline. Defaults to
            :data:`DEFAULT_TRANSFORM`, which matches training.

    Example:
        >>> predictor = Predictor()
        >>> pred = predictor.predict("photo.jpg")
        >>> pred.height, pred.distance, pred.angle
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.device = torch.device(device) if device is not None else default_device()
        resolved = resolve_model_path(model_path)
        state_dict, (self.h_mean, self.h_std, self.d_mean, self.d_std) = load_checkpoint(
            resolved, self.device
        )
        self.model = MultiTaskModel().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

    def _preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Load (if needed) and preprocess a single image into a tensor."""
        if isinstance(image, (str, Path)):
            image = Image.open(str(image))
        image = image.convert("RGB")
        return self.transform(image)

    def predict(self, image: Union[str, Path, Image.Image]) -> Prediction:
        """Predict geometry from a single image (file path or PIL image)."""
        return self.predict_batch([image])[0]

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
    ) -> List[Prediction]:
        """Predict geometry for several images at once.

        Accepts file paths and/or PIL images and returns one
        :class:`Prediction` per input, in the same order. Batching amortizes
        model overhead and is significantly faster than calling
        :meth:`predict` in a loop, especially on GPUs. For per-image fault
        tolerance over many files, use :meth:`predict_folder`.
        """
        if not images:
            return []
        x = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            reg, cls_logits = self.model(x)

        reg = reg.cpu().numpy()
        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
        classes = np.argmax(probs, axis=1)
        return [
            Prediction(
                height=float(reg[i, 0] * self.h_std + self.h_mean),
                distance=float(reg[i, 1] * self.d_std + self.d_mean),
                angle=float(CLASS_TO_ANGLE[int(classes[i])]),
                angle_probabilities=probs[i],
            )
            for i in range(len(images))
        ]

    def predict_folder(
        self,
        folder: Union[str, Path],
        recursive: bool = True,
        show_progress: bool = True,
        batch_size: int = 8,
    ) -> List[Tuple[str, Prediction]]:
        """Predict geometry for every image in a folder.

        Images are processed in batches of ``batch_size`` (default 8), which
        is considerably faster than one image at a time. Returns a list of
        ``(image_path, prediction)`` pairs in the same order as the sorted
        file list. Images that fail to load are skipped with a message on
        stderr.
        """
        files = gather_images(folder, recursive=recursive)
        batch_size = max(1, int(batch_size))
        bar = tqdm(total=len(files), desc="Predicting") if show_progress else None
        results: List[Tuple[str, Prediction]] = []
        try:
            for start in range(0, len(files), batch_size):
                chunk = files[start : start + batch_size]
                images: List[Image.Image] = []
                ok_paths: List[str] = []
                for path in chunk:
                    try:
                        img = Image.open(path)
                        img.load()
                        images.append(img)
                        ok_paths.append(path)
                    except Exception as exc:  # noqa: BLE001 - keep batch going
                        print(f"Error processing {path}: {exc}")
                try:
                    preds = self.predict_batch(images) if images else []
                except Exception as exc:  # noqa: BLE001 - fall back per image
                    print(f"Batch failed ({exc}); retrying image by image")
                    preds = []
                    for path in ok_paths:
                        try:
                            preds.append(self.predict(path))
                        except Exception as e:  # noqa: BLE001
                            print(f"Error processing {path}: {e}")
                results.extend(zip(ok_paths, preds))
                if bar:
                    bar.update(len(chunk))
        finally:
            if bar:
                bar.close()
        return results
