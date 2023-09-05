from typing import Optional
import torch
from torch import nn
from pytorch_pretrained_vit import ViT

class ModelShen(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = ViT('B_16_imagenet1k', pretrained=True, image_size=224)
        # self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone = torch.nn.Sequential()
        for layer in list(backbone.children())[:-2]:
            self.backbone.append(layer)

    def forward(self,x):
        return self.backbone(x)