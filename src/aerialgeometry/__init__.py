"""aerialgeometry: single-image drone height, distance and viewing-angle estimation.

Self-contained, pip-installable version of the ``infer.py`` research script
behind "Rectifying Geometry-Induced Similarity Distortions for Real-World
Aerial-Ground Person Re-Identification" (arXiv:2601.21405).

Quick start:

    from aerialgeometry import Predictor

    predictor = Predictor()
    pred = predictor.predict("photo.jpg")
    print(pred.height, pred.distance, pred.angle)
"""

from .model import ANGLE_TO_CLASS, CLASS_TO_ANGLE, MultiTaskModel
from .predictor import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_PATH,
    DEFAULT_TRANSFORM,
    MODEL_URL,
    Prediction,
    Predictor,
    default_device,
    gather_images,
    load_checkpoint,
    resolve_model_path,
)

__version__ = "0.1.1"

__all__ = [
    "ANGLE_TO_CLASS",
    "CLASS_TO_ANGLE",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_TRANSFORM",
    "MODEL_URL",
    "MultiTaskModel",
    "Prediction",
    "Predictor",
    "__version__",
    "default_device",
    "gather_images",
    "load_checkpoint",
    "resolve_model_path",
]
