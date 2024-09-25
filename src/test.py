from torchvision.models import densenet201, resnet50

model = densenet201(weights='IMAGENET1K_V1')

print(model)