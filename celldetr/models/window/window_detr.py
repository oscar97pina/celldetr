import torch
from ..deformable_detr import DeformableDETR, PostProcess
from ...util.misc import NestedTensor

import torch
import torch.nn.functional as F
from torch import nn
import math

from ...util import box_ops
from ...util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, interpolate, inverse_sigmoid)
from ...util.distributed import get_world_size, is_dist_avail_and_initialized
class WindowDETR(DeformableDETR):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                window_size=256, window_stride=256):
        super().__init__(backbone, transformer, num_classes, num_queries, num_feature_levels,
                         aux_loss=aux_loss, with_box_refine=with_box_refine, two_stage=two_stage)
        # set window parameters
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_overlap = (window_size - window_stride) / window_size
    
    def forward(self, samples):
        """
        This version is different from the original window detr. 
        Here, the entire tile is fed to the backbone. Then, the windows are defined on the multi-resolution backbone outputs.
        """
        if self.training:
            return super(WindowDETR, self).forward(samples)

        # 1 - obtain the number of windows
        if samples.tensors.size(-1) > self.window_size:
            num_windows = (samples.tensors.size(-1) - self.window_size) // self.window_stride + 1
        else:
            num_windows = 1
        
        # 2 - forward through backbone
        in_size = samples.tensors.size(-1)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        # 3 - split features into windows (in the batch dimension)
        new_features = []
        new_pos      = []
        for l, feat in enumerate(features):
            # obtain size of the feature map
            map_size = feat.tensors.shape[-1]
            # split the feature map into windows
            # adapt the window size and stride to the feature map size
            window_size = self.window_size * map_size // in_size
            window_stride = self.window_stride * map_size // in_size
            # extract windows
            windows = self.extract_windows(feat, window_size, window_stride)
            # append to list
            new_features.append(windows)
            new_pos.append(self.backbone[1](windows).to(windows.tensors.dtype))
        # update features and pos
        features = new_features
        pos = new_pos

        # 3 - forward though the neck
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose() #[s.shape for s in src]
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                #m = samples.mask
                m = torch.clone(masks[-1])
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        
        # make sure that forward runs with batches of size multiple to 64
        # get number of windows
        B = srcs[0].shape[0]
        # if batch size is smaller than 64 or is multiple, forward directly
        if B <= 64 or B % 64 == 0:
            outputs = self.forward_windows(srcs, masks, pos, query_embeds)
        else:
            Bw = (B // 64) * 64 
            outputs1 = self.forward_windows([s[:Bw] for s in srcs], [m[:Bw] for m in masks], [p[:Bw] for p in pos], query_embeds)
            outputs2 = self.forward_windows([s[Bw:] for s in srcs], [m[Bw:] for m in masks], [p[Bw:] for p in pos], query_embeds)
            outputs = dict(
                pred_logits=torch.cat([outputs1['pred_logits'], outputs2['pred_logits']]),
                pred_boxes=torch.cat([outputs1['pred_boxes'],   outputs2['pred_boxes']])
            )

        #hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
        
        # 4 - combine windows / predictions
        outputs = self.merge_outputs(outputs, num_windows)

        # 5 - compute window prediction mask
        if num_windows > 1:
            window_mask = self.get_window_mask(outputs)
        else:
            window_mask = torch.ones_like(outputs['pred_boxes'][...,0], dtype=torch.bool)

        # 6 - reshape predictions to match original format
        # (now we will have more queries per images)
        B, _, _, Q, L = outputs['pred_logits'].shape
        outputs['pred_logits'] = outputs['pred_logits'].view(B, -1, L)
        outputs['pred_boxes']  = outputs['pred_boxes'].view(B, -1, 4)
        outputs['window_mask'] = window_mask.view(B, -1, 1)

        # 7 - append num_windows to outputs
        outputs['num_windows'] = num_windows

        return outputs


    def old_forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # TODO: it only works with squared image tiles
        assert samples.tensors.size(-1) == samples.tensors.size(-2)

        # TODO: parallel window processing is done during inference.
        # during training, we forward the entire image, as it is expected
        # to be a smaller patch
        if self.training:
            return super(WindowDETR, self).forward(samples)

        # 1 - if input size is larger than self.window_size, extract windows
        if samples.tensors.size(-1) > self.window_size:
            num_windows = (samples.tensors.size(-1) - self.window_size) // self.window_stride + 1
            samples = self.extract_windows(samples)
        else:
            num_windows = 1

        # 2 - forward through DETR model
        outputs = self._forward_windows(samples)

        # 3 - combine windows / predictions
        outputs = self.merge_outputs(outputs, num_windows)

        # 4 - compute window prediction mask
        if num_windows > 1:
            window_mask = self.get_window_mask(outputs)
        else:
            window_mask = torch.ones_like(outputs['pred_boxes'][...,0], dtype=torch.bool)

        # 5 - reshape predictions to match original format
        # (now we will have more queries per images)
        B, _, _, Q, L = outputs['pred_logits'].shape
        outputs['pred_logits'] = outputs['pred_logits'].view(B, -1, L)
        outputs['pred_boxes']  = outputs['pred_boxes'].view(B, -1, 4)
        outputs['window_mask'] = window_mask.view(B, -1, 1)

        # 6 - append num_windows to outputs
        outputs['num_windows'] = num_windows

        return outputs

    def forward_windows(self, srcs, masks, pos, query_embeds):
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss and self.training:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage and self.training:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    def _forward_windows(self, samples):
        """
        As by creating windows we're modifying the batch size, this may lead to errors.
        For MSDeformableAttention we need to have a batch size between [1 and 64] or a multiple of 64 (128, 192, etc).
        This is due to the im2col parameter of MSDeformableAttention, which is set to 64 (hard-coded)
        To avoid probems, we'll forward by batches of 64 windows.
        """
        # get number of windows
        B = samples.tensors.shape[0]
        # if batch size is smaller than 64 or is multiple, forward directly
        if B <= 64 or B % 64 == 0:
            # run forward of the DeformableDETR
            return super(WindowDETR, self).forward(samples)

        # find the max batch size that is multiple of 64 and smaller than B
        Bw = (B // 64) * 64
        # forward the first Bw windows
        outputs = super(WindowDETR, self).forward(
            NestedTensor(samples.tensors[:Bw], samples.mask[:Bw]))
        # forward the remaining windows
        outputs_remaining = super(WindowDETR, self).forward(
            NestedTensor(samples.tensors[Bw:], samples.mask[Bw:]))
        # concatenate outputs
        outputs = dict(
            pred_logits=torch.cat([outputs['pred_logits'], outputs_remaining['pred_logits']]),
            pred_boxes=torch.cat([outputs['pred_boxes'], outputs_remaining['pred_boxes']])
        )

        """
        # get number of batches
        num_batches = (B - 1) // 64 + 1
        # forward by batches of 64 windows
        outputs = []
        for i in range(num_batches):
            # get batch of windows
            batch = NestedTensor(
                samples.tensors[i*64:(i+1)*64],
                samples.mask[i*64:(i+1)*64]
            )
            # forward
            out = super(WindowDETR, self).forward(batch)
            # append to list
            outputs.append(out)
        # concatenate outputs
        outputs = dict(pred_logits=torch.cat([out['pred_logits'] for out in outputs]),
                       pred_boxes=torch.cat([out['pred_boxes'] for out in outputs]))
        """
        # return
        return outputs
    
    def extract_windows(self, samples : NestedTensor, window_size=None, window_stride=None):
        samples.tensors = self._extract_windows(samples.tensors, window_size, window_stride)
        samples.mask    = self._extract_windows(samples.mask.unsqueeze(1), window_size, window_stride).squeeze(1)
        return samples
    
    def _extract_windows(self, x : torch.Tensor, window_size=None, window_stride=None):
        """
        Split the image into overlapped windows.
        Concat all windows along batch dimension
        (similar to window attention in SwinTransformer).
        """
        if window_size is None:
            window_size = self.window_size
        if window_stride is None:
            window_stride = self.window_stride
        B, C, H, W = x.shape
        # compute windows
        windows = x.unfold(2, window_size, window_stride).unfold(3, window_size, window_stride)
        # reshape windows
        windows = windows.contiguous().view(B, C, -1, window_size, window_size)
        # transpose windows
        windows = windows.transpose(1, 2)
        # reshape windows
        windows = windows.contiguous().view(-1, C, window_size, window_size)
        return windows
    
    def merge_outputs(self, outputs, num_windows):
        """
        Merge outputs for each image.
        Input : Dict[str:Tensor]:
            outputs['pred_logits'] of size [batch_size * num_windows, num_queries, num_classes]
            outputs['pred_boxes'] of size [batch_size * num_windows, num_queries, 4].
            outputs['pred_logits'] are in format [cx, cy, w, h] and normalized by window size. (range [0, 1])

        Output:
            outputs['pred_logits'] of size [batch_size, num_windows, num_windows, num_queries, num_classes]
            outputs['pred_boxes'] of size [batch_size, num_windows, num_windows,  num_queries, 4].
        """
        # 1 - combine bounding boxes
        pred_boxes = outputs['pred_boxes']
        pred_boxes = pred_boxes.view(-1, num_windows, num_windows, pred_boxes.size(-2), pred_boxes.size(-1))

        # 2 - combine logits
        pred_logits = outputs['pred_logits']
        pred_logits = pred_logits.view(-1, num_windows, num_windows, pred_logits.size(-2), pred_logits.size(-1))

        # 3 - return
        return dict(pred_logits=pred_logits, pred_boxes=pred_boxes)

    def get_window_mask(self, outputs):
        """
        Returns a mask selecting what are the queries that should be considered.
        This is used to remove the predictions in the borders of the overlapped windows.
        """
        drop_mask = torch.zeros_like(outputs['pred_boxes'][:,:,:,:,0], dtype=torch.bool)
        if self.window_overlap > 0.0:
            overlap = self.window_overlap
            # get centroids for each query
            cx = outputs['pred_boxes'][:,:,:,:,0]
            cy = outputs['pred_boxes'][:,:,:,:,1]
            # remove detections with a cx between 0 and overlap/2 (except for first column)
            drop_mask[:,:,1:,:] += cx[:,:,1:,:] < overlap/2
            # remove detections with a cx between 1-overlap/2 and 1 (except for last column)
            drop_mask[:,:,:-1,:] += cx[:,:,:-1,:] > 1-overlap/2
            # remove detections with a cy between 0 and overlap/2 (except for first row)
            drop_mask[:,1:,:,:] += cy[:,1:,:,:] < overlap/2
            # remove detections with a cy between 1-overlap/2 and 1 (except for last row)
            drop_mask[:,:-1,:,:] += cy[:,:-1,:,:] > 1-overlap/2
        return ~drop_mask

class WindowPostProcess(PostProcess):
    def __init__(self, method='topk', 
                 window_size=256, window_stride=256):
        super().__init__(method=method)
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_overlap = (window_size - window_stride) / window_size

    def forward(self, outputs, target_sizes):
        """
        
        """
        # TODO: now, the postprocessing is done by combining window predictions and running standard post-processing (note that with many more queries)
        # an alternative would be to postprocess windows independently and then combine, which one is better?
        
        # get logits, boxes and window masks
        pred_logits = outputs['pred_logits']
        pred_boxes  = outputs['pred_boxes']
        wind_mask   = outputs['window_mask']

        num_windows = outputs.get('num_windows', 1)
        image_size = self.window_size + (num_windows - 1) * self.window_stride

        if num_windows == 1:
            return super(WindowPostProcess, self).forward(outputs, target_sizes)

        # reshape logits, boxes and masks if needed
        if pred_logits.ndim == 3 and num_windows > 1:
            pred_logits = pred_logits.view(pred_logits.size(0), num_windows, num_windows, -1, pred_logits.size(-1))
            pred_boxes  = pred_boxes.view(pred_boxes.size(0), num_windows, num_windows, -1, pred_boxes.size(-1))
            wind_mask   = wind_mask.view(wind_mask.size(0), num_windows, num_windows, -1)
  
        # get offset of the windows
        y_off, x_off = torch.meshgrid(torch.arange(num_windows), 
                                    torch.arange(num_windows))
        y_off = y_off * self.window_stride 
        x_off = x_off * self.window_stride

        # to device
        y_off = y_off.to(pred_boxes.device)
        x_off = x_off.to(pred_boxes.device)

        # offset the boxes
        pred_boxes[...,0] = pred_boxes[...,0] * self.window_size \
                            + x_off.view(1, num_windows, num_windows, 1)
        pred_boxes[...,1] = pred_boxes[...,1] * self.window_size \
                            + y_off.view(1, num_windows, num_windows, 1)

        # scale the boxes to be on image range
        pred_boxes[...,:2] /= target_sizes[:,None,None,None,:].to(pred_boxes.device)#1000#image_size
        pred_boxes[...,2:] *= self.window_size / target_sizes[:,None,None,None,:].to(pred_boxes.device)

        # set logits of borders to -inf
        pred_logits[~wind_mask] = float('-inf')

        # reshape and create dict
        B, _, _, Q, _ = pred_boxes.shape
        L = pred_logits.size(-1)
        outputs = dict(pred_boxes=pred_boxes.view(B, -1, 4), 
                       pred_logits=pred_logits.view(B, -1, L) )

        # 3 - postprocess
        outputs = super(WindowPostProcess, self).forward(outputs, target_sizes)
        # 4 - return
        return outputs