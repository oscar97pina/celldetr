import os
import os.path as osp
import json

import numpy as np
from PIL import Image
import cv2
import scipy.io as sio

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from .base import BaseCellCOCO, DetectionWrapper

#CONSEP_NUCLEI = ['other','inflammatory','epithelial','dysplastic','fibroblast','muscle','endothelial']
CONSEP_NUCLEI = ['miscellaneous', 'inflammatory', 'epithelial', 'spindleshaped']
ACTUAL_CONSEP_NUCLEI_MAP = {1 : 1, 2 : 2, 3 : 3, 4 : 3, 5 : 4, 6 : 4, 7 : 4}

class Consep(torchvision.datasets.CocoDetection, BaseCellCOCO):
    def __init__(self, root, fold, transforms=None):
        self.root = root
        self.fold = fold
        img_folder = osp.join(root, f'{fold}', 'images')
        ann_file = osp.join(root, f'{fold}', 'annotations.json')
        super(Consep, self).__init__(img_folder, ann_file, transforms=transforms)

    @property
    def num_classes(self):
        return 4

    @property
    def class_names(self):
        return CONSEP_NUCLEI

    def image_size(self, image_id=None, idx=None):
        return torch.tensor([1024, 1024])

    def __len__(self):
        return super(Consep, self).__len__()

    def __getitem__(self, idx):
        img, tgt = super(Consep, self).__getitem__(idx)

        # empty target
        if len(tgt) == 0:
            tgt = [dict(
                        id=-1,
                        image_id=idx,
                        category_id=-1,
                        bbox=[-1, -1, -1, -1],
                        area=-1,
                        segmentation=[-1],
                        iscrowd=0,
                    )]

        return img, tgt
    
    def get_raw_image(self, image_id=None, idx=None):
        # get image id
        assert image_id is not None or idx is not None
        if image_id is None:
            image_id = self.ids[idx]
        # open image to RGB
        img = self._load_image(image_id)
        # convert to tensor
        transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize((1024, 1024),
                      interpolation=v2.InterpolationMode.BICUBIC,
                      antialias=True),
            v2.ToDtype(torch.float32, scale=True)])
        img = transforms(img)
        return img

def build_consep_dataset(cfg, split='train'):
    from .transforms import build_transforms
    # get root
    root = cfg.dataset[split].root
    num_classes = cfg.dataset[split].num_classes
    # build transforms
    transforms = build_transforms(cfg, is_train = (split=='train') )
    transforms.transforms.insert(3, v2.Resize((1024,1024), 
                                            interpolation=v2.InterpolationMode.BICUBIC,
                                            antialias=True ))
    # build dataset
    if num_classes == 1:
        dataset = DetectionWrapper(Consep)(root, cfg.dataset[split].fold,
                                        transforms=transforms)
    else:
        dataset = Consep(root, cfg.dataset[split].fold,
                    transforms=transforms)
    # wrap dataset for transforms v2
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset,
                                target_keys=('image_id','boxes','labels'))
    
    return dataset
    
def consep2coco(data_dir, fold, out_dir):
    # paths to data
    img_dir = osp.join(data_dir, fold, "images")
    lbl_dir = osp.join(data_dir, fold, "labels")

    # create output directory for images
    out_dir = osp.join(out_dir, f"{fold}")
    if not osp.exists(osp.join(out_dir, "images")):
        os.makedirs(osp.join(out_dir, "images"))

    ls_images, ls_annots = list(), list()
    instance_count = 1  # instance id starts from 1, it is accumulated for each image

    # iterate over images
    for img_idx, img_name in enumerate(os.listdir(img_dir)):
        # path to image and label
        img_path = osp.join(img_dir, img_name)
        lbl_path = osp.join(lbl_dir, img_name.split(".")[0] + ".mat")

        assert osp.exists(lbl_path)

        # proces image
        # * read image
        img = Image.open(img_path)
        # * get size of the image
        width, height = img.size # PIL img sizes are w, h
        # * save image into dst folder
        img_filename = osp.join(out_dir, "images", img_name)
        img.save(img_filename)
        # * append meta
        ls_images.append(
            dict(id=img_idx, file_name=img_filename, height=height, width=width)
        )

        # process labels
        # * read labels
        lbl = sio.loadmat(lbl_path)
        inst_map, type_map, inst_type = lbl['inst_map'], lbl['type_map'], lbl['inst_type']
        # * get uq instances
        uq_inst_ids = np.unique(inst_map)
        # * iterate for each instance
        for inst_id in uq_inst_ids:
            # avoid background
            if inst_id == 0:
                continue
            # get the mask of the instance
            inst_mask_i = inst_map == inst_id
            
            # get the coordinates of the pixels for current instance
            coords = np.where(inst_mask_i)
            
            # get the bounding box for the current instance
            xmin = int(np.min(coords[1]))
            ymin = int(np.min(coords[0]))
            xmax = int(np.max(coords[1]))
            ymax = int(np.max(coords[0]))

            # get contours from binary mask
            contours, _ = cv2.findContours(
                inst_mask_i.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # prepare contour for coco format
            contour = list()
            for p in contours[0].reshape(-1,2):
                contour.append(int(p[0]))
                contour.append(int(p[1]))
            contour = [contour]

            # get instance type
            inst_type_i = int(inst_type[int(inst_id)-1])
            inst_type_i = ACTUAL_CONSEP_NUCLEI_MAP[inst_type_i]

            # append annotations
            ls_annots.append(
                dict(
                    id=instance_count,
                    image_id=img_idx,
                    category_id=inst_type_i,
                    bbox=[xmin, ymin, xmax-xmin, ymax-ymin],
                    area=(xmax-xmin)*(ymax-ymin),
                    segmentation=contour,
                    iscrowd=0,
                )
            )
            instance_count += 1    

    # categories JSON
    categories = [dict(id=k+1, name=v) for k, v in enumerate(CONSEP_NUCLEI)]
    # prepare COCO format JSON
    coco_format_json = dict(
        images = ls_images,
        annotations = ls_annots,
        categories = categories
    )
    # save coco format json
    with open(osp.join(out_dir, "annotations.json"), 'w') as f:
        json.dump(coco_format_json, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Consep')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='data directory')
    parser.add_argument('--fold', type=str, default='train',
                        help='fold')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='output directory')
    args = parser.parse_args()
    consep2coco(args.data_dir, args.fold, args.out_dir)