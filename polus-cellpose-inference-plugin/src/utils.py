'''

Code sourced code from Cellpose repo https://github.com/MouseLand/cellpose/tree/master/cellpose

'''
import colorsys
import os
import shutil
import tempfile
from urllib.request import urlopen

import numpy as np
from scipy.ndimage import find_objects, binary_fill_holes
from tqdm import tqdm


def rgb_to_hsv(arr):
    """ Convert array from rgb to hsv color system
    Args:
        arr(float): Rgb array
    Returns:
        hsv(float): Hsv array

    """
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h, s, v), axis=-1)
    return hsv


def hsv_to_rgb(arr):
    """Convert array from hsv to rgb color system
    Args:
        hsv(float): Hsv array
    Returns:
        arr(float): Rgb array

    """
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r, g, b), axis=-1)
    return rgb


def diameters(masks):
    """ Get median 'diameter' of masks
    Args:
        masks(array): Numpy array(Ly x Lx)
    Returns:
        md(int): Median of diameter
        counts(int): Count of unique masks

    """
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts ** 0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi ** 0.5) / 2
    return md, counts ** 0.5


def normalize99(img):
    """ Normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile
    Args:
        img(array) : Numpy array that's (x  Ly x Lx x nchan)
    Returns:
        x(array) : Normalised numpy image

    """
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ Fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    Args:
        masks(array[int]): 2D or 3D array.labelled masks, 0=NO masks; 1,2,...=mask labels
        min_size(int): Default 15.minimum number of pixels per mask, can turn off with -1
    Returns:
        masks(array[int]): 2D or 3D array.masks with holes filled and masks smaller than min_size removed
    
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array' % masks.ndim)

    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:
                if msk.ndim == 3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j + 1)
                j += 1
    return masks


def download_url_to_file(url, dst, progress=True):
    """Download object at the given URL to a local path.

    Args:
        url(string): URL of the object to download
        dst(string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): Whether or not to display a progress bar to stderr

    """
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
