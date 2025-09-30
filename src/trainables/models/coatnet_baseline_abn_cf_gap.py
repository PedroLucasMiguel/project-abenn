import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class CoatNetABNCFGAP(nn.Module):
    def __init__(self, baseline_model, n_classes:int, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)

        self.stem = copy.deepcopy(baseline_model.stem)
        self.stage1 = nn.Sequential(
            copy.deepcopy(baseline_model.stages[0]),
            copy.deepcopy(baseline_model.stages[1]),
            copy.deepcopy(baseline_model.stages[2]),
        )

        self.attention_branch = nn.Sequential(
            copy.deepcopy(baseline_model.stages[3]),
            nn.AdaptiveAvgPool2d(14),
            #nn.BatchNorm2d(192),
            nn.Conv2d(768, n_classes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(),
            nn.Conv2d(n_classes, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.stage2 = copy.deepcopy(baseline_model.stages[3])
        
        self.norm = copy.deepcopy(baseline_model.norm)

        #self.gap_conv = nn.Conv2d(768, n_classes, 1)

        #self.classifier = nn.AvgPool2d(7)

        self.classifier = baseline_model.head
        self.classifier.fc = nn.Linear(768, n_classes)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = F.adaptive_avg_pool2d(grad, 1)

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        self.att = self.attention_branch(x)

        rx = x * self.att
        rx = rx + x

        rx = self.stage2(rx)
        return rx
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        self.att = self.attention_branch(x)
        rx = x * self.att
        rx = rx + x

        rx = self.stage2(rx)

        # Para o grad-cam
        #rx = F.relu(rx, inplace=True)

        if rx.requires_grad:
            h = rx.register_hook(self.activations_hook)

        rx = self.norm(rx)

        #rx = self.gap_conv(rx)

        #rx = self.classifier(rx)
        #rx = rx[:,:,0,0]

        rx = self.classifier(rx)

        return rx