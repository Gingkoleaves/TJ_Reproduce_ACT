"""ResNet visual backbone matching ACT's ``detr/models/backbone.py``."""

import torch
from torch import nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(nn.Module):
    """The fixed BatchNorm implementation used by the original ACT backbone."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + 1e-5).rsqrt()
        return x * scale + (bias - running_mean * scale)


class VisionEncoderBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, output_dim: int):
        super().__init__()
        self.backbone = IntermediateLayerGetter(backbone, return_layers={'layer4': '0'})
        self.output_dim = output_dim
        self.num_channels = output_dim

    def forward(self, x):
        return self.backbone(x)


class VisionEncoder(VisionEncoderBackbone):
    def __init__(self, name: str, position_encoder: nn.Module, pretrained=True, dilation=False):
        if name not in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            raise ValueError(f'ACT VisionEncoder only supports ResNet backbones, got {name}')
        weights = models.get_model_weights(name).DEFAULT if pretrained else None
        backbone = getattr(models, name)(
            weights=weights,
            replace_stride_with_dilation=[False, False, dilation],
            norm_layer=FrozenBatchNorm2d,
        )
        output_dim = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, output_dim)
        self.position_encoder = position_encoder

    def forward(self, x):
        feature_dict = super().forward(x)
        features, positions = [], []
        for _, feature in feature_dict.items():
            features.append(feature)
            positions.append(self.position_encoder(feature).to(feature.dtype))
        return features, positions
