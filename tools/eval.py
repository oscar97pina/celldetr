import argparse
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler

from collections import OrderedDict

import sys
sys.path.append('../celldetr')
from celldetr.util.distributed import init_distributed_mode, get_rank
from celldetr.util.misc import seed_everything
from celldetr.util.config import ConfigDict
from celldetr.data import build_dataset, build_loader
from celldetr.models import build_model
from celldetr.models.window import wrap_window_detr, WindowDETR
from celldetr.engine import evaluate_detection

def test(cfg):
    # init distributed mode
    init_distributed_mode(cfg)
    device = torch.device("cuda:" + str(cfg.gpu) if torch.cuda.is_available() else "cpu")
    
    # set seed
    seed = cfg.experiment.seed + get_rank()
    seed_everything(seed)

    # build validation and test datasets
    test_dataset = build_dataset(cfg, split='test')

    # build loaders
    test_loader  = build_loader(cfg, test_dataset, split='test')

    # build model
    model, criterion, postprocessors = build_model(cfg)
    model.to(device)
    criterion.to(device)

    # load checkpoint
    # as we are in test, we get the checkpoint from the config file (output_dir + output_name)
    path = osp.join(cfg.experiment.output_dir, cfg.experiment.output_name)
    ckpt = torch.load(path, map_location='cpu')
    ckpt = ckpt['model'] if 'model' in ckpt else ckpt
    # if loading a checkpoint from a model not wrapped with window detr, but mdel is window detr
    #if 'window' in cfg.model and not all([k.startswith('model.') for k in ckpt.keys()]):
    #    ckpt = {'model.'+k: v for k, v in ckpt.items()}
    # load checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=True)
    print(f"\t # model keys: {len(model.state_dict().keys())}, # checkpoint keys: {len(ckpt.keys())}")
    print(f"\t # missing keys: {len(missing_keys)}, # unexpected keys: {len(unexpected_keys)}")

    # distributed model
    if cfg.distributed:
        model = nn.parallel.DistributedDataParallel(model, 
                                                device_ids=[cfg.gpu])

    # evaluate
    test_stats = evaluate_detection(
        model, criterion, postprocessors, test_loader, device, thresholds=cfg.evaluation.thresholds)
    stats = {
        **{f'test_{k}': v for k, v in test_stats.items()}
    }
    print(stats)
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CellDETR')

    # config-file
    parser.add_argument('--config-file', type=str, default=None,
                        help='config file')
    # options
    # override config options with key=value pairs
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config_file is not None, "Please provide a config file."
    cfg = ConfigDict.from_file(args.config_file)
    opts = ConfigDict.from_options(args.opts)
    cfg.update(opts)
    test(cfg)