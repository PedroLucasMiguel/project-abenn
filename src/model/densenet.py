from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class DenseNet201ABENN(nn.Module):
    def __init__(self, baseline_model, n_classes:int = 2, *args, **kwargs) -> None:
        super(DenseNet201ABENN, self).__init__()

        self.growth_rate = 32

        self.att = None

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [   
                    ("first-conv", nn.Sequential(
                        baseline_model.features.conv0,
                        baseline_model.features.norm0,
                        baseline_model.features.relu0,
                        baseline_model.features.pool0,
                    )),

                    ("conv-block", nn.Sequential(
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
                    ("map-creator", nn.Sequential(
                        nn.BatchNorm2d(896),
                        nn.Conv2d(896, n_classes, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(n_classes),
                        nn.ReLU(),
                        nn.Conv2d(n_classes, 1, kernel_size=3, padding=1),
                        nn.BatchNorm2d(1),
                        nn.Sigmoid()
                    ))
                ]
            )
        )

        self.baseline = baseline_model

        
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("last-conv-block", baseline_model.features.denseblock4),
                    ("batch-norm", baseline_model.features.norm5),
                    ("relu", nn.ReLU(inplace=True)),
                    ("avg_pool", nn.AdaptiveAvgPool2d((1,1))),
                    ("flatten", nn.Flatten()),
                    ("fc", nn.Linear(1920, n_classes, bias=True))
                ]
            )
        )



    def forward(self, x:Tensor) -> Tensor:

        x = self.feature_extractor(x)

        self.att = self.attention_branch(x)

        rx = x * self.att
        rx = rx + x

        #rx = self.baseline.features.denseblock4(rx)
        #rx = self.baseline.features.norm5(rx)
        #rx = F.relu(rx, inplace=True)
        #rx = F.adaptive_avg_pool2d(rx, (1, 1))
        #rx = self.baseline.classifier(rx)

        rx = self.classifier(rx)

        return rx
    
if __name__ == "__main__":

    IMG_NAME = "cat.jpg"

    device = "cuda"
    print(f"Using {device}")

    img = Image.open(IMG_NAME).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = preprocess(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(device)

    baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    print(baseline)
    model = DenseNet201ABENN(baseline)
    model = model.to(device)

    model.eval()

    outputs = model(input_batch)
    att = model.att.detach().cpu().numpy()[0, 0, : , :]

    print(outputs)

    plt.imshow(att)
    plt.show()

    