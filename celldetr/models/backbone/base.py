import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from typing import List,Dict
import torch
from ...util.misc import NestedTensor

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, 
                 strides : List[int], 
                 num_channels : List[int]):
        super().__init__()
        self.strides = strides
        self.num_channels = num_channels
        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos