import openslide
import numpy as np
from skimage import color
from skimage.morphology import disk, binary_closing, binary_opening, remove_small_holes

def _get_slide(slide=None, filename=None):
    assert slide is not None or filename is not None,\
        "Either slide or filename must be provided"
    if slide is None:
        slide = openslide.OpenSlide(filename)
    return slide

def get_slide_bounds(slide=None, filename=None):
    slide = _get_slide(slide=slide, filename=filename)
    
    #x0 = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_X])
    #y0 = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
    #w  = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH])
    #h  = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
    x0 = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0)
    y0 = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0)
    w  = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH, slide.dimensions[0])
    h  = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT, slide.dimensions[1])
    x0, y0, w, h = int(x0), int(y0), int(w), int(h)
    return (x0, y0, w, h)

def get_slide_best_downsample(slide=None, filename=None, downsample=32):
    slide = _get_slide(slide=slide, filename=filename)
    
    lvl = slide.get_best_level_for_downsample(downsample)
    downsample = slide.level_downsamples[lvl]
    return lvl, downsample

def get_slide_thumbnail(slide=None, filename=None, downsample=32):
    slide = _get_slide(slide=slide, filename=filename)

    x0, y0, w, h = get_slide_bounds(slide=slide)

    # * get the best level for the downsample and the actual downsample
    lvl, downsample = get_slide_best_downsample(slide=slide, downsample=downsample)

    # * get the downsampled level dimensions of the bounding box
    w_lvl = int(w / downsample)
    h_lvl = int(h / downsample)

    # * read thumbnail
    thumbnail = slide.read_region((x0, y0), lvl, (w_lvl, h_lvl))
    thumbnail = thumbnail.convert("RGB")
    thumbnail = np.array(thumbnail)[:,:,:3]

    return thumbnail

# Tissue Mask
def get_hed_tissue_mask(img, h_thresh=1.0, e_thresh=0.0, d_thresh=0.5):
    img_hed = color.rgb2hed(img)
    h, e, d = img_hed[:,:,0], img_hed[:,:,1], img_hed[:,:,2]
    mask = (d<d_thresh) & ((h>h_thresh) | (e>e_thresh))
    return mask

# Smoothed Tissue Mask
def get_smoothed_hed_tissue_mask(img, h_thresh=1.0, e_thresh=0.0, d_thresh=0.5, disk_size=20):
    mask = get_hed_tissue_mask(img, h_thresh=h_thresh, e_thresh=e_thresh, d_thresh=d_thresh)
    # apply morphological operations
    mask = binary_closing(mask,  disk(disk_size))
    mask = binary_opening(mask, disk(disk_size))
    mask = remove_small_holes(mask, disk_size)
    return mask

# Return patches from mask
def list_patches(mask, 
                 downsample=1 , 
                 patch_size=256, 
                 stride=128):
    patches = list()
    # slide dimensions
    h, w = mask.shape
    h, w = int(h*downsample), int(w*downsample)
    # patch dimensions
    patch_size_ds = int(patch_size / downsample)

    for x in range(0, w-patch_size+1, stride):
        for y in range(0, h-patch_size+1, stride):
            # get the patch coordinates in downsampled space
            x_ds = int(x / downsample)
            y_ds = int(y / downsample)
            # check if the patch is in the tissue mask
            if mask[y_ds:y_ds+patch_size_ds, x_ds:x_ds+patch_size_ds].sum() > 0:
                patches.append((x, y))
    return patches
