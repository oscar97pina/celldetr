from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

#from ...util.misc import NestedTensor, is_main_process
from ...util.misc import NestedTensor
from ...util.distributed import is_main_process

from .base import BackboneBase

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
class ResNetBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 frozen_layers : List[int],
                 return_layers: List[int],
                 dilation: bool,
                 norm_layer: nn.Module):
        
        # instantiate ResNet backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"

        # set strides and channels
        all_strides = [4, 8, 16, 32]
        strides = [all_strides[l-1] for l in return_layers]
        all_channels = [256, 512, 1024, 2048]
        num_channels = [all_channels[l-1] for l in return_layers]

        # set frozen params
        frozen_layers = [f'layer{layer}' for layer in frozen_layers]
        for name, parameter in backbone.named_parameters():
            if not train_backbone or any(layer in name for layer in frozen_layers):
                parameter.requires_grad_(False)

        # set return layers
        return_layers = {f'layer{layer}' : str(idx) for idx, layer in enumerate(return_layers)}
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        super().__init__(backbone, strides, num_channels)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


def build_resnet_backbone(cfg):
    # train backbone iff if lr_backbone > 0
    train_backbone = True #cfg.optimizer.lr_backbone > 0

    # if training, some layers of the backbone can be frozen
    frozen_layers = []
    if cfg.model.backbone.frozen_stages > 0:
        frozen_layers = list(range(1, cfg.model.backbone.frozen_stages + 1))

    # normalization layer to be used (FrozenBatchNorm2d or nn.BatchNorm2d)
    norm_layer = FrozenBatchNorm2d if cfg.model.backbone.frozen_bn else nn.BatchNorm2d
    
    backbone = ResNetBackbone(
        name=cfg.model.backbone.name,
        train_backbone=train_backbone,
        frozen_layers=frozen_layers,
        return_layers=cfg.model.backbone.return_layers,
        dilation = cfg.model.backbone.dilation,
        norm_layer = norm_layer
    )

    return backbone