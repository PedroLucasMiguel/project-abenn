import torch
import torch.nn as nn

class ConvNextSmall(nn.Module):
    def __init__(self, model, n_classes:int, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.model.classifier[2] = nn.Linear(768, n_classes)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.model.features(x)
    
    def forward(self, x):
        out = self.model.features(x)

        if out.requires_grad:
            h = out.register_hook(self.activations_hook)

        out = self.model.avgpool(out)

        out = self.model.classifier(out)
        return out