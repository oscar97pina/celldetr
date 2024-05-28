from .window_detr import WindowDETR, WindowPostProcess
def wrap_window_detr(cfg, detr, postprocessors):
    # window size and stride
    wsize, wstride = cfg.model.window.size, cfg.model.window.stride
    # build window detr
    wdetr = WindowDETR(
        detr.backbone, detr.transformer, 
        num_classes=detr.num_classes, num_queries=detr.num_queries, 
        num_feature_levels=detr.num_feature_levels,
        aux_loss=detr.aux_loss, with_box_refine=detr.with_box_refine, 
        two_stage=detr.two_stage,
        window_size=wsize, window_stride=wstride
    )
    # copy weights
    wdetr.load_state_dict(detr.state_dict(), strict=True)
    # wrap postprocessors
    # TODO: it will only work with bbox postprocessor
    assert len(postprocessors) == 1 and 'bbox' in postprocessors
    wpostprocessors = {
        k:WindowPostProcess(p.method, wsize, wstride)
        for k, p in postprocessors.items()
    }
    return wdetr, wpostprocessors
