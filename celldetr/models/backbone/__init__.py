import torch
from .resnet import build_resnet_backbone
from .swin import build_swin_backbone
from .position_encoding import build_position_encoding
from .base import Joiner

def build_backbone(cfg):
    if cfg.model.backbone.name.startswith('resnet'):
        backbone = build_resnet_backbone(cfg)
    elif cfg.model.backbone.name.startswith('swin'):
        backbone = build_swin_backbone(cfg)
    else:
        raise ValueError(f"Unrecognized backbone: {cfg.model.backbone.name}")

    position_encoding = build_position_encoding(cfg)

    return Joiner(backbone, position_encoding)

def load_sd_backbone(backbone, checkpoint):
    # get model, if checkpoint is a dict
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    # keep only backbone parameters and remove 'backbone.' prefix
    #checkpoint = {k.replace('backbone.', ''): v for k, v in checkpoint.items() if 'backbone' in k}
    # keep only backbone parameters and remove 'backbone.0.body.' prefix
    # Note that Swin Transformer checkpoints from other projects have a different structure
    # therefore we first remove backbone.0.body and then backbone.0
    print(f"\t loading backbone with {len(checkpoint)} keys...")
    checkpoint = {
        k.replace('backbone.0.body.', '').replace('backbone.0.', ''): v\
        for k, v in checkpoint.items() if 'backbone' in k
    }
    # load state dict
    missing_keys, unexpected_keys = backbone[0].body.load_state_dict(checkpoint, strict=False)
    print(f"\t #backbone keys: {len(backbone[0].body.state_dict().keys())}, #checkpoint keys: {len(checkpoint.keys())}")
    print(f"\t #missing keys: {len(missing_keys)}, #unexpected keys: {len(unexpected_keys)}")