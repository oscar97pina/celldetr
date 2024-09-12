import torch

def ClassMapWrapper(base_class, map):
    """
    Map is a list of lists, where each list is a mapping of the original classes to the new classes.
    """
    class ClassMapDETR(base_class):
        def __init__(self, *args, **kwargs):
            super(ClassMapDETR, self).__init__(*args, **kwargs)
            self.num_classes = len(map)
        
        def _convert_logits(self, logits):
            # map logits to new classes
            ls_logits = list()
            for i, v in enumerate(map):
                ls_logits.append(logits[...,v].max(dim=-1)[0].unsqueeze(-1))
            logits = torch.cat(ls_logits, dim=-1)
            return logits

        def forward(self, samples):
            outputs = super(ClassMapDETR, self).forward(samples)
            outputs['pred_logits'] = self._convert_logits(outputs['pred_logits'])

            if 'aux_outputs' in outputs:
                for i in range(len(outputs['aux_outputs'])):
                    outputs['aux_outputs'][i]['pred_logits'] = self._convert_logits(outputs['aux_outputs'][i]['pred_logits'])
            
            if 'enc_outputs' in outputs:
                outputs['enc_outputs']['pred_logits'] = self._convert_logits(outputs['enc_outputs']['pred_logits'])
            
            return outputs
    
    ClassMapDETR.__name__ = 'ClassMap' + base_class.__name__
    return ClassMapDETR
            