"""Model architecture for single-image aerial geometry estimation.

Extracted from the research codebase behind
"Rectifying Geometry-Induced Similarity Distortions for Real-World
Aerial-Ground Person Re-Identification" (arXiv:2601.21405).
"""

import torch.nn as nn
from torchvision import models

#: Mapping from viewing angle (degrees) to class index.
ANGLE_TO_CLASS = {30.0: 0, 60.0: 1, 90.0: 2}
#: Mapping from class index to viewing angle (degrees).
CLASS_TO_ANGLE = {v: k for k, v in ANGLE_TO_CLASS.items()}


class MultiTaskModel(nn.Module):
    """ResNet50 backbone with a shared head and two task heads.

    Predicts, from a single image:

    * ``reg_head``  -- drone height and distance (regression, 2 outputs)
    * ``cls_head``  -- viewing angle among {30, 60, 90} degrees (3-class)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        base = models.resnet50(weights=None)
        in_feat = base.fc.in_features
        modules = list(base.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, len(ANGLE_TO_CLASS)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        feat = self.shared(x)
        reg = self.reg_head(feat)
        cls_logits = self.cls_head(feat)
        return reg, cls_logits
