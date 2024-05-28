import os
import numpy as np
import torchvision
from PIL import Image
import openslide

class FolderPatchDataset(torchvision.datasets.VisionDataset):
    """Dataset for Folder patch extraction. The dataset expects the images to be stored in a folder.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = list(sorted([f for f in os.listdir(root) if self._is_image(f)]))
        super().__init__(root=root, transform=transform)

    def _is_image(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # get image path
        img_path = os.path.join(self.root, self.imgs[idx])
        # open image
        img = Image.open(img_path).convert("RGB")
        # apply transforms
        if self.transform is not None:
            img = self.transform(img)
        # return the image
        return img, (idx,), img.size()[-2:]
    
class SlidePatchDataset(torchvision.datasets.VisionDataset):
    """Dataset for WSI patch extraction."""
    def __init__(self,
                 slide_path : str,
                 coords : list,
                 patch_size : int,
                 transform=None,):
        self.slide = openslide.OpenSlide(slide_path)
        self.coords = coords
        self.patch_size = patch_size
        super().__init__(root=None, transform=transform)
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        # get the patch coordinates
        x, y = self.coords[idx]
        # get the patch
        patch = self.slide.read_region((x, y), 0, 
                        (self.patch_size, self.patch_size)).convert("RGB")
        # apply transforms
        if self.transform is not None:
            patch = self.transform(patch)
        # return the patch
        return patch, (x, y), (self.patch_size, self.patch_size)