from torchvision.models import densenet201, resnet50, efficientnet_b0, convnext_base, convnext_small, resnext50_32x4d

from trainables.models.coatnet_baseline import *
import timm

#model = efficientnet_b0(weights='IMAGENET1K_V1')

#model = timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=True)

model = resnext50_32x4d(weights='IMAGENET1K_V1')

print(model.layer4)