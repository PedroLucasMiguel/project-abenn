import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNetGradCam(nn.Module):
    def __init__(self, model, n_classes:int, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.model.classifier = nn.Linear(1920, n_classes)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.model.features(x)
    
    def forward(self, x):
        # 1° Passa a entrada pelas camadas convolucionais
        out = self.model.features(x)
        # 2° Passa a saída das camadas por uma camda de Relu e pooling
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        if out.requires_grad:
            # 3° Hook que salva o gradiente dessa saída quando backwards() é chamado
            h = out.register_hook(self.activations_hook)
        # 4° Termina de passar a entrada para o resto da rede
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)
        return out