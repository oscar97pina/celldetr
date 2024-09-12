import os
import numpy as np
import torchvision
from PIL import Image
import openslide
from torchvision.transforms import v2

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
                 patch_size  : int,
                 patch_mpp : float = 0.25,
                 transform=None,):
        self.slide = openslide.OpenSlide(slide_path)
        self.coords = coords
        self.patch_size = patch_size
        self.patch_mpp  = patch_mpp

        # get the mpp of the slide at level 0
        slide_0_mpp = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
        # downsample factor
        self.patch_downsample = round(patch_mpp / slide_0_mpp)
        # get the slide level for that downsample
        self.slide_level = self.slide.get_best_level_for_downsample(self.patch_downsample)
        # get the actual downsample we obtain when reading from the slide
        self.slide_downsample = self.slide.level_downsamples[self.slide_level]
        # get the actual patch size, as the slide downsample can be different from the required patch downsample
        self.slide_patch_size = round(self.patch_size * self.patch_downsample / self.slide_downsample)
        # show warning if the slide downsample is different from the patch downsample
        if self.patch_downsample != self.slide_downsample:
            print(f"Warning: the slide downsample is different from the patch downsample. "
                  f"Whereas the inference process will work, the manual downsampling will influence the inference time.")
            # and append the resizing transform
            rsz = v2.Resize(self.patch_size,
                             interpolation=v2.InterpolationMode.BICUBIC, 
                             antialias=True,)
            if transform is not None:
                transform.transforms.append(rsz)
            else:
                transform = v2.Compose([rsz])

        super().__init__(root=None, transform=transform)

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        # get the patch coordinates
        x, y = self.coords[idx]
        # get the patch
        patch = self.slide.read_region((x, y), self.slide_level, 
                        (self.slide_patch_size, self.slide_patch_size)).convert("RGB")
        # apply transforms
        if self.transform is not None:
            patch = self.transform(patch)
        # return the patch
        return patch, (x, y), (self.patch_size, self.patch_size)