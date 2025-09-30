from torchvision.models import densenet201, resnet50, efficientnet_b0, convnext_base, convnext_small, resnext50_32x4d, maxvit_t

from trainables.models.coatnet_baseline import *
import timm
from trainables.models.uniformer_baseline import uniformer_xs
from trainables.models.coatnet_baseline import CoatNetGradCAM

# model = CoatNetGradCAM(timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=True), n_classes=2)


model = uniformer_xs()
#model = efficientnet_b0(weights='IMAGENET1K_V1')

#model = timm.create_model('coatnet_0_rw_224.sw_in1k', pretrained=True)

#model = maxvit_t(weights='IMAGENET1K_V1')

print(model)