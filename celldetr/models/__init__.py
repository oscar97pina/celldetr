from .backbone import build_backbone, load_sd_backbone

def build_model(cfg):
    # 1 - build backbone
    backbone = build_backbone(cfg)

    # 2 - build detection transformer
    if cfg.model.name == 'deformable_detr':
        from .deformable_detr import build_deformable_detr
        model, criterion, postprocessor = build_deformable_detr(cfg, backbone)
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not implemented")
    
    # 3 - wrap in window detr if needed
    if 'window' in cfg.model:
        from .window import wrap_window_detr
        model, postprocessor = wrap_window_detr(cfg, model, postprocessor)

    # 4.1 - classmap
    if cfg.model.has('classmap') and cfg.model.classmap:
        from .detection import wrap_classmap_detr
        classmap = [[0]] + cfg.model.classmap
        model, criterion = wrap_classmap_detr(model, criterion, classmap)
    # 4.2 - detection wrapper
    elif cfg.model.has('detection') and cfg.model.detection and cfg.model.num_classes > 1:
        from .detection import wrap_detection_only_detr
        model, criterion = wrap_detection_only_detr(model, criterion)
    
    return model, criterion, postprocessor

def load_state_dict(cfg, model):
    import torch
    
    if cfg.model.has('checkpoint'):
        # load file
        checkpoint = torch.load(cfg.model.checkpoint, map_location='cpu')

        # get model, if checkpoint is a dict
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        # get number of keys in checkpoint
        num_ckpt_keys = len(checkpoint)

        # load checkpoint of the entire model
        if cfg.model.name == 'deformable_detr':
            print("Loading checkpoint for deformable DETR...")
            from .deformable_detr import load_sd_deformable_detr

            # if we have a checkpoint for backbone, remove backbone now
            if cfg.model.backbone.has('checkpoint'):
                # remove backbone from checkpoint
                print("\t removing backbone from checkpoint...")
                checkpoint = {k: v for k, v in checkpoint.items() if 'backbone' not in k}

                # track number of keys removed
                print(f"\t {num_ckpt_keys - len(checkpoint)} keys removed from checkpoint...")
                num_ckpt_keys = len(checkpoint)

                # remove neck if number of levels and channels are different
                model_levels = model.num_feature_levels
                ckpt_levels  = len([k for k in checkpoint.keys() if k.startswith('input_proj') and k.endswith('0.weight')])
                model_channels = [model.input_proj[i][0].in_channels for i in range(model_levels)]
                ckpt_channels  = [checkpoint["input_proj.{}.0.weight".format(i)].size(1) for i in range(ckpt_levels)]
                if model_levels != ckpt_levels or not all([m==c for m,c in zip(model_channels, ckpt_channels)]):
                    print("\t removing neck from checkpoint...")
                    checkpoint = {k: v for k, v in checkpoint.items() if 'input_proj' not in k}

                    print(f"\t {num_ckpt_keys - len(checkpoint)} keys removed from checkpoint...")
                    num_ckpt_keys = len(checkpoint)
                
            load_sd_deformable_detr(model, checkpoint)
        else:
            raise NotImplementedError(f"Model {cfg.model.name} not implemented")

    # if specific checkpoint for backbone, load it
    if cfg.model.backbone.has('checkpoint'):
        print("Loading checkpoint for backbone...")

        # load file
        checkpoint = torch.load(cfg.model.backbone.checkpoint, map_location='cpu')
        # get model, if checkpoint is a dict
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        # load state dict
        load_sd_backbone(model.backbone, checkpoint)