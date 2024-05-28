import torch
import math
"""
Fix for DistributedSampler that doesn't add duplicates.
In validation, the last batch may be smaller than the others, and the DistributedSampler will add duplicates to make it the same size.
Extracted from https://github.com/RUCAIBox/RecBole/pull/1872
"""
class DistributedSamplerNoDuplicate(torch.utils.data.DistributedSampler):
    """ A distributed sampler that doesn't add duplicates. Arguments are the same as DistributedSampler """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # some ranks may have less samples, that's fine
            #if rank >= len(self.dataset) % self.num_replicas:
            #    self.num_samples -= 1
            self.num_samples = int(math.ceil((len(self.dataset)-self.rank) * 1.0 / self.num_replicas))
            self.total_size = len(self.dataset)