import os
import os.path as osp

import numpy as np
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from .base import BaseCellCOCO, DetectionWrapper

PANNUKE_TISSUE = ['Adrenal_gland', 'Bile-duct', 'Bladder', 'Breast', 'Cervix', 'Colon',
 'Esophagus', 'HeadNeck', 'Kidney', 'Liver', 'Lung', 'Ovarian', 'Pancreatic',
 'Prostate', 'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus']

PANNUKE_NUCLEI = ['neoplastic', 'inflammatory', 'connective', 'necrosis', 'epithelial']

class Pannuke(torchvision.datasets.CocoDetection, BaseCellCOCO):
    def __init__(self, root, fold, transforms=None):
        self.root = root

        fold = fold if isinstance(fold, str) else f'fold{fold}'
        self.fold = fold

        img_folder = osp.join(root, fold, 'images')
        ann_file = osp.join(root,   fold, 'annotations.json')
        super(Pannuke, self).__init__(img_folder, ann_file, transforms=transforms)
    
    @property
    def num_classes(self):
        return 5
    
    @property
    def class_names(self):
        return PANNUKE_NUCLEI
    
    def image_size(self, image_id=None, idx=None):
        return torch.tensor([256, 256])
    
    def __len__(self):
        return super(Pannuke, self).__len__()

    def __getitem__(self, idx):
        img, tgt = super(Pannuke, self).__getitem__(idx)

        # invalid target
        if len(tgt) > 0:
            tgt = [t for t in tgt if t['area']>0 and len(t['segmentation'][0])>4] # remove invalid targets

        # empty target
        if len(tgt) == 0:
            # this is a dummy target that will be removed by transforms
            tgt = [dict(
                        id=-1,
                        image_id=idx,
                        category_id=-1,
                        bbox=[-1, -1, -1, -1],
                        area=255*255,
                        segmentation=[[0,0,0,255,255,255,255,0]],
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

def build_pannuke_dataset(cfg, split='train'):
    from .transforms import build_transforms
    root = cfg.dataset[split].root
    num_classes = cfg.dataset[split].num_classes
    # build transforms
    transforms = build_transforms(cfg, is_train = (split=='train') )
    # build dataset
    if num_classes == 1:
        dataset = DetectionWrapper(Pannuke)(root, cfg.dataset[split].fold,
                               transforms=transforms)
    else:
        dataset = Pannuke(root, cfg.dataset[split].fold,
                               transforms=transforms)
    # wrap dataset for transforms v2
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset,
                                target_keys=('image_id','boxes','labels'))
    return dataset

def pannuke2coco(data_dir, fold, out_dir):
    import json
    import numpy as np
    import cv2
    from PIL import Image
    f"""Converts the Pannuke dataset to COCO format.
    Args:
        data_dir (str): path to the data directory.
        fold (int): fold number.
        out_dir (str): path to the output directory.
    """
    print("Converting Pannuke to COCO format...")
    # paths to data
    img_path = osp.join(data_dir, f"Fold {fold}", "images", f"fold{fold}", "images.npy")
    mask_path = osp.join(data_dir, f"Fold {fold}", "masks", f"fold{fold}", "masks.npy")
    #img_path = osp.join(data_dir, fold, "images", fold, "images.npy")
    #mask_path = osp.join(data_dir, fold, "masks", fold, "masks.npy")

    # load images and masks
    images = np.load(img_path)
    masks = np.load(mask_path)[:,:,:,:-1] # ignore background mask

    # create output directory for images
    out_dir = osp.join(out_dir, f"fold{fold}")
    if not osp.exists(osp.join(out_dir, "images")):
        os.makedirs(osp.join(out_dir, "images"))
    
    ls_images, ls_annots = list(), list()
    instance_count = 1  # instance id starts from 1, it is accumulated for each image
    
    # iterate over images
    for idx in range(images.shape[0]):
        # prepare image name (4-digit number for filename)
        filename = "im{:04d}.png".format(idx)

        # get image
        image_i = images[idx]
        # save image (in RGB format)
        #cv2.imwrite(osp.join(out_dir, "images", filename), image_i)
        Image.fromarray(image_i.astype(np.uint8)).save(osp.join(out_dir, "images", filename))

        # prepare image json
        height, width = image_i.shape[:2]
        ls_images.append(
            dict(id=idx, file_name=filename, height=height, width=width)
        )

        # prepare masks
        mask_i = masks[idx]
        # get all annotations
        for lbl in range(mask_i.shape[-1]):
            uq_instance_ids = np.unique(mask_i[:, :, lbl])[1:]
            for instance_id in uq_instance_ids:
                # get the coordinates of pixels for current instance
                coords = np.where(mask_i[:, :, lbl] == instance_id)
                # get the bounding box for the current instance
                xmin = int(np.min(coords[1]))
                ymin = int(np.min(coords[0]))
                xmax = int(np.max(coords[1]))
                ymax = int(np.max(coords[0]))

                # get binary mask for the object
                mask_i_bin = mask_i[:, :, lbl] == instance_id
                # get contours from binary mask
                contours, _ = cv2.findContours(
                    mask_i_bin.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                # convert contours to coco format (list of list of points)
                #contour = [[int(p[0]), int(p[1])] for p in contours[0].reshape(-1,2)] # only one object
                contour = list()
                for p in contours[0].reshape(-1,2):
                    contour.append(int(p[0]))
                    contour.append(int(p[1]))
                contour = [contour]
                # prepare dict
                ls_annots.append(
                    dict(
                        id=instance_count,
                        image_id=idx,
                        category_id=int(lbl+1),
                        bbox=[xmin, ymin, xmax-xmin, ymax-ymin],
                        area=(xmax-xmin)*(ymax-ymin),
                        segmentation=contour,
                        iscrowd=0,
                    )
                )
                instance_count += 1

    # prepare categories json
    categories = [dict(id=k+1, name=v) for k,v in enumerate(PANNUKE_NUCLEI) ]
    # prepare coco format json
    coco_format_json = dict(
        images = ls_images,
        annotations = ls_annots,
        categories = categories,
    )
    # save coco format json
    with open(osp.join(out_dir, "annotations.json"), 'w') as f:
        json.dump(coco_format_json, f)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--fold', type=str, default=1)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    assert args.data_dir is not None, "Please provide the path to the data directory."
    pannuke2coco(args.data_dir, args.fold, args.out_dir)