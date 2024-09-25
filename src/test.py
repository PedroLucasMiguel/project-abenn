from torchvision.models import densenet201, resnet50, efficientnet_b0

model = efficientnet_b0(weights='IMAGENET1K_V1')

print(model.features[8])