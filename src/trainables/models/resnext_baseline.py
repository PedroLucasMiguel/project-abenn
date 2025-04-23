import torch
import torch.nn as nn
import torch.functional as F

class ResNext50(nn.Module):
    def __init__(self, model, n_classes:int, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=False)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        return out
    
    def forward(self, x):
        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)

        if out.requires_grad:
            h = out.register_hook(self.activations_hook)

        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.model.fc(out)
        return out