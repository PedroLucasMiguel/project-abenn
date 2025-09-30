import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List
import torch.nn.functional as F
from torch import Tensor
import copy

class EfficientNetABNCFGAP(nn.Module):
    def __init__(self, baseline_model, n_classes:int, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)

        self.features = nn.Sequential(
            copy.deepcopy(baseline_model.features[0]),
            copy.deepcopy(baseline_model.features[1]),
            copy.deepcopy(baseline_model.features[2]),
            copy.deepcopy(baseline_model.features[3]),
            copy.deepcopy(baseline_model.features[4]),
            copy.deepcopy(baseline_model.features[5]),
            copy.deepcopy(baseline_model.features[6]),
            copy.deepcopy(baseline_model.features[7])
        )

        self.attention_branch = nn.Sequential(
            baseline_model.features[8][0],
            nn.BatchNorm2d(1280),
            nn.Conv2d(1280, n_classes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(),
            nn.Conv2d(n_classes, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.last_block = nn.Sequential(copy.deepcopy(baseline_model.features[8][0]), baseline_model.features[8][1], baseline_model.features[8][2])
        self.gap_conv = nn.Conv2d(1280, n_classes, 1)

        self.classifier = nn.AvgPool2d(7)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.features(x)

        self.att = self.attention_branch(x)
        
        rx = x * self.att
        rx = rx + x

        rx = self.last_block(rx)

        return rx
    
    def forward(self, x):
        x = self.features(x)

        self.att = self.attention_branch(x)
        
        rx = x * self.att
        rx = rx + x

        rx = self.last_block(rx)

        # Para o grad-cam
        rx = F.relu(rx, inplace=True)

        if rx.requires_grad:
            rx.register_hook(self.activations_hook)

        rx = self.gap_conv(rx)

        rx = self.classifier(rx)
        rx = rx[:,:,0,0]

        return rx