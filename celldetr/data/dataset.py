import os
import os.path as osp
import torch
import bisect
import torch
import torchvision
from torchvision.transforms import v2
from .base import BaseCellCOCO, DetectionWrapper
from .transforms import build_transforms

def build_cell_dataset(cfg, split='train'):
    # assert the cfg provides a root or ann_file and img_folder
    assert cfg.dataset[split].has('root') or\
                (cfg.dataset[split].has('ann_file') and\
                 cfg.dataset[split].has('img_folder'))
    
    # get img_folder and ann_file
    # * if provided, take them
    if cfg.dataset[split].has('ann_file') and cfg.dataset[split].has('img_folder'):
        img_folder = cfg.dataset[split].img_folder
        ann_file = cfg.dataset[split].ann_file
    # * if not, do it from the root
    else:
        # get root
        root = cfg.dataset[split].root       
        # get img_folder and ann_file
        img_folder = osp.join(root, 'images')
        ann_file = osp.join(root, 'annotations.json')

    # number of classes
    num_classes = cfg.dataset[split].num_classes

    # create transforms
    transforms = build_transforms(cfg, is_train = (split=='train') )

    # build dataset
    if num_classes == 1:
        dataset = DetectionWrapper(CellDataset)(img_folder, ann_file,
                                                transforms=transforms)
    else:
        dataset = CellDataset(img_folder, ann_file, transforms=transforms)

    # wrap dataset for transforms v2
    dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset,
                                target_keys=('image_id','boxes','labels'))

    return dataset

class CellDataset(torchvision.datasets.CocoDetection, BaseCellCOCO):
    def __init__(self, img_folder, ann_file, transforms=None):
        super(CellDataset, self).__init__(img_folder, ann_file, transforms=transforms)
    
    @property
    def num_classes(self):
        return len(self.coco.cats)
    
    @property
    def class_names(self):
        # get class names (category name) sorted by category id
        cat_ids = sorted(self.coco.cats.keys())
        return [self.coco.cats[cat_id]['name'] for cat_id in cat_ids]
    
    def __getitem__(self, idx):
        img, tgt = super(CellDataset, self).__getitem__(idx)

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
    
    def image_size(self, image_id=None, idx=None):
        # get image id
        assert image_id is not None or idx is not None
        if image_id is None:
            image_id = self.ids[idx]
        return torch.tensor([self.coco.imgs[image_id]['height'],
                             self.coco.imgs[image_id]['width']])
    
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

class SubsetCellDataset(torch.utils.data.Subset, BaseCellCOCO):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
    
    @property
    def num_classes(self):
        return self.dataset.num_classes
    
    @property
    def class_names(self):
        return self.dataset.class_names
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)
    
    def image_size(self, image_id=None, idx=None):
        # get oiriginal index
        if idx is not None:
            idx = self.indices[idx]
        return self.dataset.image_size(image_id=image_id, idx=idx)

    def get_raw_image(self, image_id=None, idx=None):
        # get oiriginal index
        if idx is not None:
            idx = self.indices[idx]
        return self.dataset.get_raw_image(image_id=image_id, idx=idx)
    
class ConcatCellDataset(torch.utils.data.ConcatDataset, BaseCellCOCO):
    def __init__(self, datasets):

        # ensure that all datasets have the same num_classes
        num_classes = datasets[0].num_classes
        for d in datasets[1:]:
            assert d.num_classes == num_classes, "All datasets should have the same number of classes"
        super().__init__(datasets)
    
    @property
    def num_classes(self):
        return self.datasets[0].num_classes
    
    @property
    def class_names(self):
        return self.datasets[0].class_names
    
    def get_idxs(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx
    
    def image_size(self, image_id=None, idx=None):
        assert idx is not None, "idx should be provided"
        didx, sidx = self.get_idxs(idx)
        return self.datasets[didx].image_size(image_id=image_id, idx=sidx)
    
    def get_raw_image(self, image_id=None, idx=None):
        assert idx is not None, "idx should be provided"
        didx, sidx = self.get_idxs(idx)
        return self.datasets[didx].get_raw_image(image_id=image_id, idx=sidx)