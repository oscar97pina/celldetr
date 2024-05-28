from .detection_wrapper import DetectionWrapper

def wrap_detection_only_detr(detr, criterion):
    # build window detr
    kwargs = dict(
        num_classes=detr.num_classes, num_queries=detr.num_queries,
        num_feature_levels=detr.num_feature_levels,
        aux_loss=detr.aux_loss, with_box_refine=detr.with_box_refine,
        two_stage=detr.two_stage,
    )
    if hasattr(detr,'window_size') and hasattr(detr,'window_stride'):
        kwargs['window_size'] = detr.window_size
        kwargs['window_stride'] = detr.window_stride

    detdetr = DetectionWrapper(detr.__class__)(
            detr.backbone, detr.transformer, 
            **kwargs
    )
    # copy weights
    detdetr.load_state_dict(detr.state_dict(), strict=True)
    # num classes of criterion is 1
    criterion.num_classes = 1
    return detdetr, criterion