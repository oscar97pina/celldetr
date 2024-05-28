import torch
from ..backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .deformable_detr import DeformableDETR, PostProcess, SetCriterion
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from .matcher import build_matcher

def build_deformable_detr(cfg, backbone):
    # the num_classes of the model refers to max_obj_id (i.e. 5 for pannuke) +1
    num_classes = cfg.model.num_classes + 1
    transformer = build_deforamble_transformer(cfg)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=cfg.model.num_queries,
        num_feature_levels=cfg.model.num_feature_levels,
        aux_loss=cfg.model.aux_loss,
        with_box_refine=cfg.model.with_box_refine,
        two_stage=cfg.model.two_stage,
    )

    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    
    assert cfg.matcher.name == 'HungarianMatcher', "Currently only HungarianMatcher is supported"
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.loss.class_coef, 
                   'loss_bbox': cfg.loss.bbox_coef,
                   'loss_giou': cfg.loss.giou_coef}
    
    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #weight_dict["loss_mask"] = args.mask_loss_coef
        #weight_dict["loss_dice"] = args.dice_loss_coef
    
    # TODO this is a hack
    if cfg.model.aux_loss:
        aux_weight_dict = {}
        for i in range(cfg.model.transformer.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #losses += ["masks"]
    
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=cfg.loss.focal_alpha)
    postprocessors = {'bbox': PostProcess(cfg.model.postprocess)}
    
    if 'masks' in cfg.model and cfg.model.masks:
        raise NotImplementedError("Mask head not implemented")
        #postprocessors['segm'] = PostProcessSegm()
        #if args.dataset_file == "coco_panoptic":
        #    is_thing_map = {i: i <= 90 for i in range(201)}
        #    postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

def load_sd_deformable_detr(model, checkpoint):
    # get model, if checkpoint is a dict
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    # remove class_embed from checkpoint
    checkpoint = {k: v for k, v in checkpoint.items() if 'class_embed' not in k}

    print(f"\t loading deformable detr with {len(checkpoint)} keys...")
    # load state dict
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print(f"\t # model keys: {len(model.state_dict().keys())}, # checkpoint keys: {len(checkpoint.keys())}")
    print(f"\t # missing keys: {len(missing_keys)}, # unexpected keys: {len(unexpected_keys)}")