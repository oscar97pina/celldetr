import argparse
import os
import os.path as osp
import time
import pandas as pd

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
        # coordinates are provided by the user
        print("Pre-processing | Using provided coordinates in a CSV file. Make sure it does not contain header.")
        # read the coordinates, convert to list of tuples of ints
        coords = pd.read_csv(cfg.inference.preprocessing.coords, sep=None, engine='python', header=None)
        coords = coords.astype(int).values.tolist()
        print(f"Pre-processing | Found {len(coords)} patches.")
        return coords
    elif cfg.inference.has('preprocessing'):
        # coordinates are not provided by the user
        print("Pre-processing | Extracting coordinates from the slide.")
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
        # get the patch coordinates
        # * input patch size is in the desired mpp, but we need coordinates in the level 0 space
        # ** get the mpp of the slide
        slide_mpp, _       = oslide.get_slide_mpp(slide=slide)
        # ** get the downsample factor to scale patch size and stride
        patch_downsample   = round(cfg.inference.patch.mpp / slide_mpp)
        # ** get the patch coordinates
        coords = oslide.list_patches(mask,
                                    downsample=cfg.inference.preprocessing.downsample,
                                    patch_size=cfg.inference.patch.size * patch_downsample,
                                    stride=cfg.inference.patch.stride * patch_downsample)
        print(f"Pre-processing | Found {len(coords)} patches.")
        return coords

def build_inference_dataset(cfg):
    input_path  = cfg.inference.input_path
    output_dir = cfg.inference.output_dir

    # if output path does not exist, create it
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # ensure output path is a directory
    assert osp.isdir(output_dir), \
        "Output path must be a directory."
    
    if osp.isdir(input_path):
        # create the dataset
        dataset = FolderPatchDataset(input_path, 
                        transform=build_inference_transforms(cfg))
    else:
        # for slide dataset, rescale is not supported in inference
        if cfg.transforms.has('rescale'):
            raise ValueError("Rescale as input configuration is not supported in inference on WSIs, set transforms.rescale=None."+\
                "\n If your model works at a given resolution (ie. 0.50mpp), specify it in cfg.inference.patch.mpp.")

        # get the patch coordinates
        coords = get_patch_coords(cfg)
        # create the dataset
        dataset = SlidePatchDataset(input_path, 
                        coords, 
                        patch_size=cfg.inference.patch.size, 
                        patch_mpp=cfg.inference.patch.mpp,
                        transform=build_inference_transforms(cfg))
    return dataset

def build_inference_transforms(cfg):
    import torchvision.transforms.v2 as v2
    transforms = [v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)]
    if cfg.transforms.has('rescale'):
        from celldetr.data.transforms import Rescale
        transforms.append(Rescale(cfg.transforms.rescale, 
                            antialias=True,
                            interpolation=v2.InterpolationMode.BICUBIC))
    transforms.append(v2.Normalize(mean=cfg.transforms.normalize.mean, 
                     std=cfg.transforms.normalize.std))
    
    return v2.Compose(transforms)

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
    ###########################################################################
    # Setup
    ###########################################################################

    # init distributed mode
    init_distributed_mode(cfg)
    device = torch.device("cuda:" + str(cfg.gpu) if torch.cuda.is_available() else "cpu")

    # time tracking
    t0 = time.time()

    # create dataset
    dataset = build_inference_dataset(cfg)

    # create loader
    cfg.loader.infer.shuffle=False
    cfg.loader.infer.drop_last=False
    loader = build_loader(cfg, dataset, split='infer', collate_fn=collate_fn)

    # load pre-trained model
    model, postprocessors = load_inference_model(cfg)
    model = model.to(device)

    # time tracking 
    t1 = time.time()

    ###########################################################################
    # Inference
    ###########################################################################

    # inference
    model.eval()
    ls_preds_gpu, ls_preds_cpu = list(), list()
    cpu_interval = 10
    with torch.no_grad():
        for i, (samples, position, sizes) in enumerate(loader):
            # move to device
            position = position.to(device)
            sizes   = sizes.to(device)
            samples = samples.to(device)

            # forward and postprocessing
            outputs = model(samples)
            outputs = postprocessors['bbox'](outputs, sizes)
            
            # prepare outputs  
            for i, out in enumerate(outputs):
                # convert boxes to cx, cy, w, h format
                out['boxes'] = box_xyxy_to_cxcywh(out['boxes'])
                # mask predictions with score < threshold
                mask = out['scores'] > cfg.inference.postprocessing.threshold
                outputs[i]['labels'] = out['labels'][mask]
                outputs[i]['boxes'] = out['boxes'][mask]
                outputs[i]['scores'] = out['scores'][mask]
                outputs[i]['position'] = position[i,:]

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
    
    ###########################################################################
    # Post-processing
    ###########################################################################
    # post-processing, only in the main process
    if not is_main_process():
        return
      
    # create a dict of predictions with the position as keys
    # * this will remove duplicated patches in distributed settings
    predictions = {tuple(out["position"].tolist()) : out for out in ls_preds_cpu}

    # if slide dataset and level>0, convert the predictions to level 0
    if isinstance(dataset, SlidePatchDataset) and dataset.patch_downsample > 1:
        scale = dataset.patch_downsample
        for pos, pred in predictions.items():
            predictions[pos]['boxes'] *= scale

    # deal with edge cases (only for SlidePatchDataset)
    if isinstance(dataset, SlidePatchDataset):
        # get patch size, stride and overlap at level 0
        psize  = cfg.inference.patch.size    * dataset.patch_downsample
        pstride = cfg.inference.patch.stride * dataset.patch_downsample
        poverlap = psize - pstride
        # iterate over patches, remove predictions in the outer margin
        for pos, pred in predictions.items():
            # get the neighbors of the current patch
            has_left_neigh = (pos[0]-pstride, pos[1]) in predictions
            has_right_neigh = (pos[0]+pstride, pos[1]) in predictions
            has_top_neigh = (pos[0], pos[1]-pstride) in predictions
            has_bottom_neigh = (pos[0], pos[1]+pstride) in predictions
            # if neigh, remove the elements in the outer margin of size stride/2
            mask = torch.ones_like(pred["labels"], dtype=torch.bool)
            if has_left_neigh:
                mask = mask & (pred["boxes"][:,0] > poverlap/2)
            if has_right_neigh:
                mask = mask & (pred["boxes"][:,0] < psize - poverlap/2)
            if has_top_neigh:
                mask = mask & (pred["boxes"][:,1] > poverlap/2)
            if has_bottom_neigh:
                mask = mask & (pred["boxes"][:,1] < psize - poverlap/2)
            # update the predictions
            predictions[pos]["labels"] = pred["labels"][mask]
            predictions[pos]["boxes"]  = pred["boxes"][mask]
            predictions[pos]["scores"] = pred["scores"][mask]
    
    # save predictions
    qupath = cfg.inference.postprocessing.has('qupath') and cfg.inference.postprocessing.qupath
    # * if folder patch dataset, save one csv file for each prediction
    if isinstance(dataset, FolderPatchDataset):
        for pos, pred in predictions.items():
            # get the name of the image file
            idx   = pos[0]
            fname = dataset.imgs[idx]
            fname = fname[:fname.rfind('.')] # without extension
            # save the prediction
            save_pred_df(create_pred_df(pred, qupath=qupath),
                        osp.join(cfg.inference.output_dir, fname),
                        qupath=qupath)
    else:
        # * if slide patch dataset, save one csv file for all predictions
        # ** offset the predictions by the patch position
        for pos in predictions:
            predictions[pos]['boxes'][:,0] += pos[0]
            predictions[pos]['boxes'][:,1] += pos[1]
        # ** concatenate all predictions
        pred = {
            'boxes'  : torch.cat([pred['boxes'] for pred in predictions.values()], dim=0),
            'labels' : torch.cat([pred['labels'] for pred in predictions.values()], dim=0),
            'scores' : torch.cat([pred['scores'] for pred in predictions.values()], dim=0)
        }
        # ** get the slide name
        slide_name = osp.basename(cfg.inference.input_path)
        slide_name = slide_name[:slide_name.rfind('.')]
        # ** save the prediction
        save_pred_df(create_pred_df(pred, qupath=qupath),
                    osp.join(cfg.inference.output_dir, slide_name),
                    qupath=qupath)

    # time tracking
    t3 = time.time()
    # print time
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print(f'Load Time: {t1-t0:.2f}')
    print(f'Inference Time: {t2-t1:.2f}')
    print(f'Post Processing Time: {t3-t2:.2f}')
    print(f'Total Time: {t3-t0:.2f}')

def create_pred_df(pred, qupath=False):
    if qupath:
        return pd.DataFrame({'x':     pred['boxes'][:,0].tolist(),
                             'y':     pred['boxes'][:,1].tolist(),
                             'name':  pred['labels'].tolist(),
                             'color': [256]*pred['labels'].size(0)})
    return pd.DataFrame({'cx': pred['boxes'][:,0].tolist(),
                         'cy': pred['boxes'][:,1].tolist(),
                         'w':  pred['boxes'][:,2].tolist(),
                         'h':  pred['boxes'][:,3].tolist(),
                         'score': pred['scores'].tolist(),
                         'label': pred['labels'].tolist()})

def save_pred_df(df, filename, qupath=False):
    if qupath:
        df.to_csv(filename+".tsv", sep='\t', index=False)
    else:
        df.to_csv(filename+".csv", index=False)

def _check_infer_cfg(cfg):
    # check the config file
    assert cfg.has('inference'), "Please provide an inference configuration."
    assert cfg.inference.has('input_path'), "Please provide an input path."
    assert cfg.inference.has('output_dir'), "Please provide an output directory."

    # check the preprocessing if input_path is a slide
    if not osp.isdir(cfg.inference.input_path):
        if cfg.inference.has('preprocessing') and cfg.inference.preprocessing.has('coords'):
                assert osp.isfile(cfg.inference.preprocessing.coords), "Coords file must be a CSV file."
                # in this case, patch size and stride is mandatory
                assert cfg.inference.has('patch'), "Please provide a patch configuration."
                assert cfg.inference.patch.has('size'), "Please provide a patch size."
                assert cfg.inference.patch.has('stride'), "Please provide a patch stride."
                # patch mpp is optional, default is 0.25 but it'll depend on the model the user's employing
                cfg.inference.patch.mpp = cfg.inference.patch.get('mpp', 0.25)
        else:
            # check the preprocessing, which is optional
            if not cfg.inference.has('preprocessing'):
                cfg.inference.preprocessing = ConfigDict()
            # set default values for non-provided preprocessing parameters
            cfg.inference.preprocessing.downsample = cfg.inference.preprocessing.get('downsample', 32)
            cfg.inference.preprocessing.h_thresh = cfg.inference.preprocessing.get('h_thresh', 1.0)
            cfg.inference.preprocessing.e_thresh = cfg.inference.preprocessing.get('e_thresh', 0.0)
            cfg.inference.preprocessing.d_thresh = cfg.inference.preprocessing.get('d_thresh', 0.5)
            cfg.inference.preprocessing.disk_size = cfg.inference.preprocessing.get('disk_size', 20)
            # patch configuration is optional
            if not cfg.inference.has('patch'):
                cfg.inference.patch = ConfigDict()
            cfg.inference.patch.size = cfg.inference.patch.get('size', 1024)
            cfg.inference.patch.stride = cfg.inference.patch.get('stride', 960)
            cfg.inference.patch.mpp = cfg.inference.patch.get('mpp', 0.25)
    # check the postprocessing, which is optional
    if not cfg.inference.has('postprocessing'):
        cfg.inference.postprocessing = ConfigDict()
    cfg.inference.postprocessing.threshold = cfg.inference.postprocessing.get('threshold', 0.4)
    cfg.inference.postprocessing.qupath = cfg.inference.postprocessing.get('qupath', False)

    return cfg 

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
    cfg = _check_infer_cfg(cfg)
    print(cfg)
    main(cfg)

    """            
    if is_main_process():
        # deal with edge cases (only for SlidePatchDataset)
        if isinstance(dataset, SlidePatchDataset):
            # get all patch positions
            all_positions = set([tuple(p["position"].tolist()) for p in ls_preds_cpu])
            # get patch size, stride and overlap
            psize  = cfg.inference.patch.size
            pstride = cfg.inference.patch.stride
            poverlap = psize - pstride
            # iterate over patches, remove predictions in the outer margin
            for out in ls_preds_cpu:
                # get position of current patch
                position = tuple(out["position"].tolist())
                # get the neighbors of the current patch
                has_left_neigh = (position[0]-pstride, position[1]) in all_positions
                has_right_neigh = (position[0]+pstride, position[1]) in all_positions
                has_top_neigh = (position[0], position[1]-pstride) in all_positions
                has_bottom_neigh = (position[0], position[1]+pstride) in all_positions
                # if neigh, remove the elements in the outer margin of size stride/2
                mask = torch.ones_like(out["labels"], dtype=torch.bool)
                if has_left_neigh:
                    mask = mask & (out["boxes"][:,0] > poverlap/2)
                if has_right_neigh:
                    mask = mask & (out["boxes"][:,0] < psize - poverlap/2)
                if has_top_neigh:
                    mask = mask & (out["boxes"][:,1] > poverlap/2)
                if has_bottom_neigh:
                    mask = mask & (out["boxes"][:,1] < psize - poverlap/2)
                # update the predictions
                out["labels"] = out["labels"][mask]
                out["boxes"] = out["boxes"][mask]
                out["scores"] = out["scores"][mask]

        # save predictions
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
                with open(osp.join(cfg.inference.output_dir, fname + ".json"), 'w') as f:
                    json.dump(pred, f)
        else:
            # * if slide patch dataset, save one json file for all predictions
            with open(cfg.inference.output_dir, 'w') as f:
                json.dump(preds, f)
    """