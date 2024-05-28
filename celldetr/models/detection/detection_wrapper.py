import torch

def DetectionWrapper(base_class):
    class DetectionDETR(base_class):
        def __init__(self, *args, **kwargs):
            super(DetectionDETR, self).__init__(*args, **kwargs)
            self.num_classes = 1
            
        def _convert_logits(self, logits):
            # convert logits to 1 single class (detection only)
            # to do so, we keep the 0th class and take the max of the rest (2 logits)
            logits_1 = logits[...,1:].max(dim=-1)[0].unsqueeze(-1)
            logits = torch.cat([logits[...,:1], logits_1], dim=-1)
            return logits
        
        def forward(self, samples):
            outputs = super(DetectionDETR, self).forward(samples)
            # convert logits to 1 single class (detection only)
            # to do so, we keep the 0th class and take the max of the rest (2 logits)
            outputs['pred_logits'] = self._convert_logits(outputs['pred_logits'])

            if 'aux_outputs' in outputs:
                for i in range(len(outputs['aux_outputs'])):
                    outputs['aux_outputs'][i]['pred_logits'] = self._convert_logits(outputs['aux_outputs'][i]['pred_logits'])
            
            if 'enc_outputs' in outputs:
                outputs['enc_outputs']['pred_logits'] = self._convert_logits(outputs['enc_outputs']['pred_logits'])
            
            return outputs

    
    DetectionDETR.__name__ = 'Detection' + base_class.__name__
    return DetectionDETR