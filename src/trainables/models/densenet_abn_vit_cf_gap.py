from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from matplotlib import pyplot as plt


class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

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
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
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
        self.conv = nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_drop_activation = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        self.attn_drop_activation = attn.detach() # "Forward hook"

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DenseNet201ABNVITGAP(nn.Module):
    def __init__(self, baseline_model, n_classes: int = 2, freeze_training: bool = False, *args, **kwargs) -> None:
        super(DenseNet201ABNVITGAP, self).__init__()

        if freeze_training:
            print("DenseNet201ABNN - Training freezed")
            for param in baseline_model.parameters():
                param.requires_grad = False

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

        self.attention_branch = Attention(896, 7)

        self.last_conv_block = baseline_model.features.denseblock4
        self.las_bn = baseline_model.features.norm5
        self.gap_conv = nn.Conv2d(1920, n_classes, 1)

        self.classifier = nn.AvgPool2d(7)

        self.gradients = None

    def gradients_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.feature_extractor(x)
        self.att = x.flatten(2) #16, 896, 49
        self.att = self.att.reshape(self.att.shape[0], self.att.shape[2], self.att.shape[1])
        self.att = self.attention_branch(self.att)#16, 896, 49
        self.att = self.attention_branch.attn_drop_activation

        result = torch.eye(self.att.shape[-1]) #49x49
        attention_heads_fused = self.att.min(axis=1)[0] #16, 49

        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1)*0.9), -1, False)
        indices = indices[indices != 0]
        flat[0, indices] = 0

        I = torch.eye(attention_heads_fused.size(-1))
        a = (attention_heads_fused + 1.0*I)/2

        final_result = torch.zeros(a.shape[0], a.shape[1], a.shape[2])
        i = 0
        for b in a:
            final_result[i,:,:] = torch.matmul((b / b.sum(dim=-1)), result)
            i+=1

        self.att = final_result.mean(dim=1)
        self.att = self.att.reshape(final_result.shape[0], 1, 7, 7)
        
        rx = x * self.att
        rx = rx + x

        rx = self.last_conv_block(rx)

        return rx

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        self.att = x.flatten(2) #16, 896, 49
        self.att = self.att.reshape(self.att.shape[0], self.att.shape[2], self.att.shape[1])
        self.att = self.attention_branch(self.att)#16, 896, 49
        self.att = self.attention_branch.attn_drop_activation

        result = torch.eye(self.att.shape[-1]) #49x49
        attention_heads_fused = self.att.min(axis=1)[0] #16, 49

        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1)*0.9), -1, False)
        indices = indices[indices != 0]
        flat[0, indices] = 0

        I = torch.eye(attention_heads_fused.size(-1))
        a = (attention_heads_fused + 1.0*I)/2

        final_result = torch.zeros(a.shape[0], a.shape[1], a.shape[2])
        i = 0
        for b in a:
            final_result[i,:,:] = torch.matmul((b / b.sum(dim=-1)), result)
            i+=1

        self.att = final_result.mean(dim=1)
        self.att = self.att.reshape(final_result.shape[0], 1, 7, 7)
        # self.att = self.att.reshape(self.att.shape[0], 896, 7, 7)
        # self.att = self.attention_branch(self.att)
        #self.att = self.attention_branch(x)

        '''
            Notas para o Predo do futuro
            o attention_heads_fused está saindo no formato [batch_size, imagem_flataned]
            temos que pegar cada um desses batches e calcular o valor de attention rollout para eles
            armazenando em um novo tensor de formato [batch_size, 1 (por ser uma imagem para cada elemento no batch), 7, 7]
        '''
        rx = x * self.att
        rx = rx + x

        rx = self.last_conv_block(rx)
        rx = self.las_bn(rx)

        # Para o grad-cam
        rx = F.relu(rx, inplace=True)

        if rx.requires_grad:
            rx.register_hook(self.gradients_hook)

        rx = self.gap_conv(rx)

        rx = self.classifier(rx)
        rx = rx[:, :, 0, 0]

        return rx
