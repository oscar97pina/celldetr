import argparse
import os
import os.path as osp
import json
import time

import openslide
from skimage import color

import torch

import sys
sys.path.append('../celldetr')
from celldetr.util.distributed import init_distributed_mode, save_on_master, is_main_process, get_rank, all_gather
from celldetr.util.config import ConfigDict
from celldetr.util.misc import nested_tensor_from_tensor_list
from celldetr.util.box_ops import box_xyxy_to_cxcywh
from celldetr.util import oslide
from celldetr.data import build_loader
from celldetr.data.wsi import SlidePatchDataset, FolderPatchDataset
from celldetr.models import build_model

def get_patch_coords(cfg):
    if cfg.inference.has('preprocessing') and \
        cfg.inference.preprocessing.has('coords'):
        assert osp.isfile(cfg.inference.preprocessing.coords) and\
            cfg.inference.preprocessing.coords.endswith(".json"),\
            "Coords file must be a JSON file."
        with open(cfg.inference.preprocessing.coords, 'r') as f:
            coords = json.load(f)
        return coords
    elif cfg.inference.has('preprocessing'):
        # read the wsi with openslide
        slide = openslide.OpenSlide(cfg.inference.input_path)
        # get the slide bounds and downsample
        x0, y0, w, h = oslide.get_slide_bounds(slide=slide)
        _, downsample = oslide.get_slide_best_downsample(slide=slide,
                                                        downsample=cfg.inference.preprocessing.downsample)
        cfg.inference.preprocessing.downsample = downsample
        # get a thumbnail
        img = oslide.get_slide_thumbnail(slide=slide, 
                                        downsample=cfg.inference.preprocessing.downsample)
        # get the tissue mask
        mask = oslide.get_smoothed_hed_tissue_mask(img,
                                                cfg.inference.preprocessing.h_thresh, 
                                                cfg.inference.preprocessing.e_thresh, 
                                                cfg.inference.preprocessing.d_thresh,
                                                cfg.inference.preprocessing.disk_size)
        coords = oslide.list_patches(mask,
                                    downsample=cfg.inference.preprocessing.downsample,
                                    patch_size=cfg.inference.patch.size,
                                    stride=cfg.inference.patch.stride)
        return coords

def build_inference_dataset(cfg):
    input_path  = cfg.inference.input_path
    output_path = cfg.inference.output_path
    if osp.isdir(input_path):
        assert osp.isdir(output_path),\
            "Output path must be a directory."
        dataset = FolderPatchDataset(input_path, 
                        transform=build_inference_transforms(cfg))
    else:
        assert output_path.endswith(".json"),\
            "Output path must be a JSON file."
        coords = get_patch_coords(cfg)
        dataset = SlidePatchDataset(input_path, 
                        coords, 
                        cfg.inference.patch.size, 
                        transform=build_inference_transforms(cfg))
    return dataset

def build_inference_transforms(cfg):
    import torchvision.transforms.v2 as v2
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cfg.transforms.normalize.mean, 
                     std=cfg.transforms.normalize.std)
    ])
    return transforms

def load_inference_model(cfg):
    # build model
    model, _, postprocessors = build_model(cfg)

    # load checkpoint
    # as we are in inference, we get the checkpoint from the config file (output_dir + output_name)
    path = osp.join(cfg.experiment.output_dir, cfg.experiment.output_name)
    ckpt = torch.load(path, map_location='cpu')
    
    # load checkpoint
    ckpt = ckpt['model'] if 'model' in ckpt else ckpt

    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=True)
    print(f"\t # model keys: {len(model.state_dict().keys())}, # checkpoint keys: {len(ckpt.keys())}")
    print(f"\t # missing keys: {len(missing_keys)}, # unexpected keys: {len(unexpected_keys)}")

    return model, postprocessors

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    batch[1] = torch.tensor(batch[1])
    batch[2] = torch.tensor(batch[2])
    return batch

def main(cfg):
    # init distributed mode
    init_distributed_mode(cfg)
    device = torch.device("cuda:" + str(cfg.gpu) if torch.cuda.is_available() else "cpu")

    # time tracking
    t0 = time.time()

    # create dataset
    dataset = build_inference_dataset(cfg)
    print(f"Processing {len(dataset)} patches")

    # create loader
    loader = build_loader(cfg, dataset, split='infer', collate_fn=collate_fn)
    
    # load pre-trained model
    model, postprocessors = load_inference_model(cfg)
    model = model.to(device)

    # time tracking 
    t1 = time.time()

    # inference
    model.eval()
    ls_preds_gpu, ls_preds_cpu = list(), list()
    cpu_interval = 10
    with torch.no_grad():
        for i, (samples, position, sizes) in enumerate(loader):
            position = position.to(device)
            samples = samples.to(device)
            sizes   = sizes.to(device)

            # forward and postprocessing
            outputs = model(samples)
            outputs = postprocessors['bbox'](outputs, sizes)
            
            # prepare outputs  
            for i, out in enumerate(outputs):
                # add position
                out['position'] = position[i,:]
                # convert boxes to cx, cy, w, h format
                out['boxes'] = box_xyxy_to_cxcywh(out['boxes'])
                # mask predictions with score < threshold
                mask = out['scores'] > cfg.inference.postprocessing.threshold
                out['labels'] = out['labels'][mask]
                out['boxes'] = out['boxes'][mask]
                out['scores'] = out['scores'][mask]

            # gather outputs from all processes
            outputs  = all_gather(outputs) # list of list of dict
            
            # move to cpu every cpu_interval iterations
            if is_main_process():
                ls_preds_gpu.append( outputs )
                if i % cpu_interval == 0 or i == len(loader)-1:
                    # send predictions to cpu
                    ls_preds_cpu.extend([{k:v.cpu() for k,v in out.items()}\
                                         for batch_outputs in ls_preds_gpu \
                                            for rank_outputs in batch_outputs \
                                                for out in rank_outputs])
                    # clear gpu memory of predictions
                    ls_preds_gpu = list()
    
    # send remaining elements
    if is_main_process() and len(ls_preds_gpu) > 0:
        ls_preds_cpu.extend([{k:v.cpu() for k,v in out.items()}\
                                for batch_outputs in ls_preds_gpu \
                                    for rank_outputs in batch_outputs \
                                        for out in rank_outputs])

    # time tracking
    t2 = time.time()
        
    # save predictions
    if is_main_process():
        # create dict of predictions
        positions, preds = set(), list()
        for out in ls_preds_cpu:
            patch = dict()
            # position of the current patch
            patch['position'] = out['position'].tolist()
            # check if the position is already in the list
            if tuple(patch['position']) in positions:
                continue
            # add position to the list
            positions.add(tuple(patch['position']))
            # nuclei of the current patch
            patch['nuclei'] = list()
            for i in range(out['labels'].size(0)):
                # append prediction
                patch['nuclei'].append({
                    'label' : out['labels'][i].item(),
                    'bbox' : out['boxes'][i].tolist(),
                    'score' : out['scores'][i].item()
                })
            preds.append(patch)

        # save predictions
        # * if folder patch dataset, save one json file for each prediction
        if isinstance(dataset, FolderPatchDataset):
            for pred in preds:
                # get the name of the image file
                idx   = pred['position'][0] # it's a tuple
                fname = dataset.imgs[idx]
                fname = fname[:fname.rfind('.')] # without extension
                # save the prediction
                with open(osp.join(cfg.inference.output_path, fname + ".json"), 'w') as f:
                    json.dump(pred, f)
        else:
            # * if slide patch dataset, save one json file for all predictions
            with open(cfg.inference.output_path, 'w') as f:
                json.dump(preds, f)

        # time tracking
        t3 = time.time()
        # print time
        print(f'Number of GPUs: {torch.cuda.device_count()}')
        print(f'Load Time: {t1-t0:.2f}')
        print(f'Inference Time: {t2-t1:.2f}')
        print(f'Post Processing Time: {t3-t2:.2f}')
        print(f'Total Time: {t3-t0:.2f}')

        
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
    print(cfg)
    main(cfg)