import torch
import torch.nn as nn
import torch.nn.functional as F

class CoatNetGradCAM(nn.Module):
    def __init__(self, baseline_model, n_classes:int, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)
        self.model = baseline_model
        self.model.head.fc = nn.Linear(768, n_classes)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = F.adaptive_avg_pool2d(grad, 1)

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        out = self.model.stem(x)
        out = self.model.stages[0](out)
        out = self.model.stages[1](out)

        out = self.model.stages[2](out)
        out = self.model.stages[3](out)
        return out
    
    def forward(self, x):
        out = self.model.stem(x)
        out = self.model.stages[0](out)
        out = self.model.stages[1](out)

        out = self.model.stages[2](out)
        out = self.model.stages[3](out)

        if out.requires_grad:
            h = out.register_hook(self.activations_hook)

        out = self.model.norm(out)
        out = self.model.head(out)

        return out