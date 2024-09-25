import torch
import torch.nn as nn

class ResNetGradCAM(nn.Module):
    def __init__(self, model, n_classes:int, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        print(model)
        self.model.fc = nn.Linear(2048, n_classes)

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
        return self.model.layer4(out)
    
    def forward(self, x):
        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)

        out = self.model.avgpool(out)

        if out.requires_grad:
            h = out.register_hook(self.activations_hook)

        out = torch.flatten(out, 1)
        out = self.model.fc(out)
        return out