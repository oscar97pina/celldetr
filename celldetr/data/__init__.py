from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from torchvision import datasets

from .pannuke import build_pannuke_dataset
from .consep import build_consep_dataset
from .monuseg import build_monuseg_dataset
from .dataset import build_cell_dataset
from ..util.misc import nested_tensor_from_tensor_list

def build_dataset(cfg, split='train'):
    if cfg.dataset[split].name == 'pannuke':
        dataset = build_pannuke_dataset(cfg, split=split)
    elif cfg.dataset[split].name == 'consep':
        dataset = build_consep_dataset(cfg, split=split)
    elif cfg.dataset[split].name == 'monuseg':
        dataset = build_monuseg_dataset(cfg, split=split)
    elif cfg.dataset[split].name == 'cell':
        dataset = build_cell_dataset(cfg, split=split)
    else:
        raise ValueError(f'Unknown dataset: {cfg.dataset[split].name}')
    
    return dataset

#collate_fn = lambda batch : tuple(zip(*batch))
def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return batch

def build_loader(cfg, dataset, split='train', collate_fn=collate_fn):
    _loader_cfg = cfg.loader[split]
    
    # create sampler
    sampler = None
    if cfg.distributed:
        if split in ['train','val','infer']:
            sampler = DistributedSampler(dataset, 
                                     shuffle=_loader_cfg.shuffle,
                                     num_replicas=cfg.world_size,
                                     rank=cfg.rank)
        else:
            from .loader import DistributedSamplerNoDuplicate
            sampler = DistributedSamplerNoDuplicate(dataset, 
                                     shuffle=_loader_cfg.shuffle,
                                     num_replicas=cfg.world_size,
                                     rank=cfg.rank)
    else:
        sampler = RandomSampler(dataset) if _loader_cfg.shuffle else SequentialSampler(dataset)
    # create data loader
    loader = DataLoader(dataset, sampler=sampler,
                        batch_size=_loader_cfg.batch_size,
                        num_workers=_loader_cfg.num_workers,
                        drop_last=_loader_cfg.drop_last,
                        collate_fn=collate_fn, pin_memory=split=="infer")
    # pin memory only for inference as if done in train or val, converts tv_tensors to standard torch tensors.
    return loader