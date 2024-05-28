import os
import os.path as osp

import numpy as np
from PIL import Image
import cv2
from xml.etree import ElementTree as ET
import json

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from .base import BaseCellCOCO

class MonusegDetection(torchvision.datasets.CocoDetection, BaseCellCOCO):
    def __init__(self, root, fold, transforms=None):
        self.root = root
        self.fold = fold
        img_folder = osp.join(root, f'{fold}', 'images')
        ann_file = osp.join(root, f'{fold}', 'annotations.json')
        super(MonusegDetection, self).__init__(img_folder, ann_file, transforms=transforms)

    @property
    def num_classes(self):
        return 1

    @property
    def class_names(self):
        return ["nuclei"]

    def image_size(self, image_id=None, idx=None):
        return torch.tensor([1000, 1000])

    def __len__(self):
        return super(MonusegDetection, self).__len__()

    def __getitem__(self, idx):
        img, tgt = super(MonusegDetection, self).__getitem__(idx)

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
            v2.ToDtype(torch.float32, scale=True)])
        img = transforms(img)
        return img

def build_monuseg_dataset(cfg, split='train'):
    from .transforms import build_transforms
    # get root
    root = cfg.dataset[split].root
    # build transforms
    transforms = build_transforms(cfg, is_train = (split=='train') )
    # build dataset
    dataset = MonusegDetection(root, cfg.dataset[split].fold,
                    transforms=transforms)
    # wrap dataset for transforms v2
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset,
                                target_keys=('image_id','boxes','labels'))
    
    return dataset

def monuseg2coco(data_dir, fold, out_dir):
    # get directories
    img_dir = osp.join(data_dir, fold, "images")
    lbl_dir = osp.join(data_dir, fold, "annotations")

    # create output directory for images
    out_dir = osp.join(out_dir, fold)
    if not osp.exists(osp.join(out_dir, "images")):
        os.makedirs(osp.join(out_dir, "images"))

    ls_images, ls_annots = list(), list()
    instance_count = 1  # instance id starts from 1, it is accumulated for each image

    # iterate over images
    for img_idx, img_name in enumerate(os.listdir(img_dir)):
        # print
        print(f"Processing image {img_name}...")

        # path to image and label
        img_path = osp.join(img_dir, img_name)
        lbl_path = osp.join(lbl_dir, img_name.split(".")[0] + ".xml")

        assert osp.exists(lbl_path), f"{lbl_path} does not exists"

        # proces image
        # * read image
        img = Image.open(img_path)
        # * get size of the image
        width, height = img.size # PIL img sizes are w, h
        # * save image into dst folder
        img_filename = osp.join(out_dir, "images", img_name.replace(".tif",".png"))
        img.save(img_filename)
        # * append meta
        ls_images.append(
            dict(id=img_idx, file_name=img_filename, height=height, width=width)
        )

        # process labels
        tree = ET.parse(lbl_path)
        tree_root = tree.getroot()
        
        for annotation_xml in tree_root.findall("Annotation"):
            # get all regions (cell nuclei)
            regions = annotation_xml.findall("Regions/Region")
            for region in regions:
                # get the coordinates
                coords = [ (float(v.attrib['X']), float(v.attrib['Y'])) for v in region.findall("Vertices/Vertex") ]
                coords = np.array(coords)
        
                # get the bounding box for the current instance
                xmin, ymin = np.min(coords, axis=0)
                xmax, ymax = np.max(coords, axis=0)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
                # get the contour in coco format
                contour = list()
                for p in coords:
                    contour.append(int(p[0]))
                    contour.append(int(p[1]))
                contour = [contour]
        
                # append annotations
                ls_annots.append(
                    dict(
                        id=instance_count,
                        image_id=img_idx,
                        category_id=1,
                        bbox=[xmin, ymin, xmax-xmin, ymax-ymin],
                        area=(xmax-xmin)*(ymax-ymin),
                        segmentation=contour,
                        iscrowd=0,
                    )
                )
                instance_count += 1    

    # categories JSON
    categories = [dict(id=1, name="nuclei")]
    # prepare COCO format JSON
    coco_format_json = dict(
        images = ls_images,
        annotations = ls_annots,
        categories = categories
    )
    # save coco format json
    with open(osp.join(out_dir, "annotations.json"), 'w') as f:
        json.dump(coco_format_json, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--fold', type=str, default="train")
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    assert args.data_dir is not None, "Please provide the path to the data directory."
    monuseg2coco(args.data_dir, args.fold, args.out_dir)