from collections import OrderedDict
from typing import Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import cv2

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class DenseNet201ABENN(nn.Module):
    def __init__(self, baseline_model, n_classes:int = 2, *args, **kwargs) -> None:
        super(DenseNet201ABENN, self).__init__()

        self.growth_rate = 32

        self.att = None

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [   
                    ("first_conv", nn.Sequential(
                        baseline_model.features.conv0,
                        baseline_model.features.norm0,
                        baseline_model.features.relu0,
                        baseline_model.features.pool0,
                    )),

                    ("conv_block", nn.Sequential(
                        baseline_model.features.denseblock1,
                        baseline_model.features.transition1,
                        baseline_model.features.denseblock2,
                        baseline_model.features.transition2,
                        baseline_model.features.denseblock3,
                        baseline_model.features.transition3,
                    )),
                ]
            )
        )

        self.attention_branch = nn.Sequential(
            OrderedDict(
                [
                    ("map_creator", nn.Sequential(
                        _DenseBlock(32, 896, 4, 32, 0, False),
                        nn.BatchNorm2d(1920),
                        nn.Conv2d(1920, n_classes, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(n_classes),
                        nn.ReLU(),
                        nn.Conv2d(n_classes, 1, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(1),
                        nn.Sigmoid()
                    ))
                ]
            )
        )
        
        self.last_conv_block = baseline_model.features.denseblock4

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("batch_norm", baseline_model.features.norm5),
                    ("relu", nn.ReLU(inplace=True)),
                    ("avg_pool", nn.AdaptiveAvgPool2d((1,1))),
                    ("flatten", nn.Flatten()),
                    ("fc", nn.Linear(1920, n_classes, bias=True))
                ]
            )
        )

        self.gradients = None

    def gradients_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.feature_extractor(x)
        self.att = self.attention_branch(x)

        rx = x * self.att
        rx = rx + x

        rx = self.last_conv_block(rx)

        return rx

    def forward(self, x:Tensor) -> Tensor:

        x = self.feature_extractor(x)

        self.att = self.attention_branch(x)

        rx = x * self.att
        rx = rx + x

        rx = self.last_conv_block(rx)

        # Para o grad-cam
        rx = F.relu(rx, inplace=True)
        rx = F.adaptive_avg_pool2d(rx, (1,1))

        # TODO - Não acho que isso vai rolar :/
        if rx.requires_grad:
            rx.register_hook(self.gradients_hook)

        rx = self.classifier(rx)

        return rx, self.att
    
if __name__ == "__main__":

    IMG_NAME = "cat.jpg"

    device = "cuda"
    print(f"Using {device}")

    img = Image.open(IMG_NAME).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    img = preprocess(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(device)

    baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    
    model = DenseNet201ABENN(baseline)
    model = model.to(device)
    print(model)

    model.eval()

    outputs = model(input_batch)
    att = model.att.detach().cpu().numpy()

    cam_normalized = np.zeros((att.shape[0], att.shape[2], att.shape[3]))

    for i in range(att.shape[0]):
        s = np.sum(att[i,0,:,:])
        cam_normalized[i,:,:] = np.divide(att[i,0,:,:], s)

    ce = np.sum(np.multiply(cam_normalized,np.log(cam_normalized)))
    print(10*ce)
    print(cam_normalized)

    #print(outputs)

    #plt.imshow(att)
    #plt.show()

    