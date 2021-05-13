# Code sourced from https://github.com/MouseLand/cellpose/tree/master/cellpose

import os, tempfile, shutil
from tqdm import tqdm
from urllib.request import urlopen

import numpy as np

def download_url_to_file(url, dst, progress=True):
    """ Download object at the given URL to a local path.
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
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

def diameters(masks):
    """ Get median 'diameter' of masks
    Args:
        masks(array[float]): Label
    Return:
        md(float): Median of labels
        counts(float): The indices of the first occurrences of the unique values

    """
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts ** 0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi ** 0.5) / 2
    return md, counts ** 0.5

