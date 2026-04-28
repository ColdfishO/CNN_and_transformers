import torch.nn as nn
from timm import create_model

class HierarchicalResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_model(
            "resnet34",
            pretrained=False,
            num_classes=0
        )
        self.head_coarse = nn.Linear(self.backbone.num_features, 20)
        self.head_fine = nn.Linear(self.backbone.num_features, 100)

    def forward(self, x):
        features = self.backbone(x)
        coarse_logits = self.head_coarse(features)
        fine_logits = self.head_fine(features)
        return coarse_logits, fine_logits