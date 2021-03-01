'''

Code sourced  code  from Cellpose repo  https://github.com/MouseLand/cellpose/tree/master/cellpose

'''
from scipy.ndimage import find_objects, binary_fill_holes
import numpy as np
import colorsys


def rgb_to_hsv(arr):
    """Convert array from rgb to hsv color system
    Args:
        arr(float): Rgb array
    Returns:
        hsv(float): hsv array

    """
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h,s,v), axis=-1)
    return hsv

def hsv_to_rgb(arr):
    """Convert array from hsv to rgb color system
    Args:
        hsv(float): hsv array
    Returns:
        arr(float): Rgb array

    """
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb

def diameters(masks):
    """ Fet median 'diameter' of masks
    Args:
        masks(array): numpy array(Ly x Lx)
    Returns:
        md(int): Median of diameter
        counts(int): Count of unique masks

    """
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5)/2
    return md, counts**0.5

def normalize99(img):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile
    Args:
        img(array) : numpy array that's (x  Ly x Lx x nchan)
    Returns:
        x(array) :  Normalised numpy image
    """
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ Fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    Args:
        masks(array[int]): 2D or 3D array.labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
        min_size(int):  default 15.minimum number of pixels per mask, can turn off with -1
    Returns:
        masks(array[int]):  2D or 3D array.masks with holes filled and masks smaller than min_size removed,
        0=NO masks; 1,2,...=mask labels,size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:    
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks