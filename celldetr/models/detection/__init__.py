from .detection_wrapper import DetectionWrapper
from .classmap_wrapper import ClassMapWrapper

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

def wrap_classmap_detr(detr, criterion, map):
        # build class map detr
    kwargs = dict(
        num_classes=detr.num_classes, num_queries=detr.num_queries,
        num_feature_levels=detr.num_feature_levels,
        aux_loss=detr.aux_loss, with_box_refine=detr.with_box_refine,
        two_stage=detr.two_stage,
    )
    if hasattr(detr,'window_size') and hasattr(detr,'window_stride'):
        kwargs['window_size'] = detr.window_size
        kwargs['window_stride'] = detr.window_stride

    mapdetr = ClassMapWrapper(detr.__class__, map)(
                            detr.backbone, detr.transformer, 
                            **kwargs
    )
    # copy weights
    mapdetr.load_state_dict(detr.state_dict(), strict=True)
    # num classes is len(map)
    detr.num_classes = len(map)
    criterion.num_classes = len(map)

    return mapdetr, criterion