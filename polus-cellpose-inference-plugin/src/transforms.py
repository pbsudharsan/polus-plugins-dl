# Code sourced code from Cellpose repo https://github.com/MouseLand/cellpose/tree/master/cellpose

import warnings
import cv2
import numpy as np


def _taper_mask(ly=224, lx=224, sig=7.5):
    """ Taper edges of tiles

    This function tapers edges of tiles.

    Args:
        ly(int): mask shape[0]
        lx(int): mask shape[1]
        sig(float): sigma
    Returns:
        mask(array): Mask after tapered

    """

    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1 / (1 + np.exp((xm - (bsize / 2 - 20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[bsize // 2 - ly // 2: bsize // 2 + ly // 2 + ly % 2,
           bsize // 2 - lx // 2: bsize // 2 + lx // 2 + lx % 2]
    return mask


def unaugment_tiles(y, unet=False):
    """ Reverse test-time augmentations for averaging

    This function unaugments a augmented tile.

    Args:
        y(array[float32]): Array that's ntiles_y x ntiles_x x chan x Ly x Lx where chan = (dY, dX, cell prob)
        unet(bool): Whether or not unet output or cellpose output
    Returns:
        y(array): Unaugmented tiles

    """

    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j % 2 == 0 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, :]
                if not unet:
                    y[j, i, 0] *= -1
            elif j % 2 == 1 and i % 2 == 0:
                y[j, i] = y[j, i, :, :, ::-1]
                if not unet:
                    y[j, i, 1] *= -1
            elif j % 2 == 1 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, ::-1]
                if not unet:
                    y[j, i, 0] *= -1
                    y[j, i, 1] *= -1
    return y


def average_tiles(y, ysub, xsub, Ly, Lx):
    """ Average results of network over tiles

    This function averages the result of network per tile.

    Args:
        y(array[float32]): [ntiles x nclasses x bsize x bsize]output of cellpose network for each tile
        ysub(list) : List of arrays with start and end of tiles in Y of length ntiles
        xsub(list) : List of arrays with start and end of tiles in X of length ntiles
        Ly(int) : Size of pre-tiled image in Y (may be larger than original image if image size is less than bsize)
        Lx(int) : Size of pre-tiled image in X (may be larger than original image if image size is less than bsize)
    Returns:
        yf(array[float32]): Network output averaged over tiles

    """

    Navg = np.zeros((Ly, Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf


def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """ Convert images to tiles for feeding to the network

    This function makes tiles of image to run at test-time if augmented, tiles are flipped and tile_overlap=2.
    Tiles are augmented following ways:
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Args:
        imgi(array[float32]) : Array that's nchan x Ly x Lx
        bsize(float) : Default 224. Size of tiles
        augment(bool) : Default False. Flip tiles and set tile_overlap=2.
        tile_overlap(float): Default 0.1. Fraction of overlap of tiles
    Returns:
        IMG(array[float32]) : Array that's ntiles x nchan x bsize x bsize
        ysub(list) : List of arrays with start and end of tiles in Y of length ntiles
        xsub(list): List of arrays with start and end of tiles in X of length ntiles

    """

    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize - Ly, Lx))), axis=1)
            Ly = bsize
        if Lx < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize - Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)
        ysub = []
        xsub = []
        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, :]
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = IMG[j, i, :, :, ::-1]
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)
        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]
    return IMG, ysub, xsub, Ly, Lx


def normalize99(img):
    """ Normalize image

    Normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile.

    Args:
        img(array) : Numpy array that's (x  Ly x Lx x nchan).
    Returns:
        x(array) : Normalised numpy image.

    """

    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X


def reshape(data, channels=[0, 0], chan_first=False):
    """ Reshape data using channels

    Function to reshape based on channel to segment.

    Args:
        data(array) : Numpy array that's (Z x ) Ly x Lx x nchan. If data.ndim==8 and data.shape[0]<8, assumed to be nchan x Ly x Lx.
        channels[list] : List of int of length 2 (optional, default [0,0]).
        invert(bool) : Invert intensities.
    Returns:
        data(array) : Numpy array that's (Z x ) Ly x Lx x nchan (if chan_first==False).

    """

    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[:, :, np.newaxis]
    elif data.shape[0] < 8 and data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))
    # use grayscale image
    if data.shape[-1] == 1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0] == 0:
            data = data.mean(axis=-1)
            data = np.expand_dims(data, axis=-1)
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0] - 1]
            if channels[1] > 0:
                chanid.append(channels[1] - 1)
            data = data[..., chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[..., i]) == 0.0:
                    if i == 0:
                        warnings.warn("Chan to seg' has value range of ZERO")
                    else:
                        warnings.warn(
                            "'Chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
            if data.shape[-1] == 1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim == 4:
            data = np.transpose(data, (3, 0, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))
    return data


def normalize_img(img, axis=-1, invert=False):
    """ Normalize input image

    Function to normalize each channel of the image so that so that 0.0=1st percentile and 1.0=99th
    percentile of image intensities and optional inversion.

    Args:
        img(array): Input image.
        axis(int): Channel axis to loop over for normalization.
    Returns:
        img(array[float32]): Normalized image of same size.

    """

    if img.ndim < 3:
        raise ValueError('Image needs to have at least 3 dimensions')
    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if np.ptp(img[k]) > 0.0:
            img[k] = normalize99(img[k])
            if invert:
                img[k] = -1 * img[k] + 1
    img = np.moveaxis(img, 0, axis)
    return img


def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR):
    """ Resize image for computing flows / unresize for computing dynamics

    This function is to resize/unresize for computing flows/ unresize for computing dynamics.

    Args:
        img0(array): Image of size [y x x x nchan]
        Ly(int): Resize shape
        Lx(int): Resize shape
        rsz(float): Resize coefficient(s) for image; if Ly is None then rsz is used
        interpolation(cv2 interp method): Type of interpolation. Default cv2.INTER_LINEAR
    Returns:
        imgs(array): Resized image of size [Ly x Lx x nchan]

    """

    if Ly is None and rsz is None:
        raise ValueError('Must give size to resize to or factor to use for resizing')
    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        Ly = int(img0.shape[-3] * rsz[-2])
        Lx = int(img0.shape[-2] * rsz[-1])
    if img0.ndim == 4:
        imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i, img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs


def pad_image_ND(img0, div=16, extra=1):
    """ Image padding for the network

    This function pads images so that the its dimensions are a multiple of 16.

    Args:
        img0(array): Image of size [nchan (x Lz) x Ly x Lx]
        div(int): Default 16
    Returns:
        I(array): Padded image
        ysub(array[int]): yrange of pixels in I corresponding to img0
        xsub(array[int]): xrange of pixels in I corresponding to img0

    """

    Lpad = int(div * np.ceil(img0.shape[-2] / div) - img0.shape[-2])
    xpad1 = extra * div // 2 + Lpad // 2
    xpad2 = extra * div // 2 + Lpad - Lpad // 2
    Lpad = int(div * np.ceil(img0.shape[-1] / div) - img0.shape[-1])
    ypad1 = extra * div // 2 + Lpad // 2
    ypad2 = extra * div // 2 + Lpad - Lpad // 2
    if img0.ndim > 3:
        pads = np.array([[0, 0], [0, 0], [xpad1, xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0, 0], [xpad1, xpad2], [ypad1, ypad2]])
    I = np.pad(img0, pads, mode='constant')
    Ly, Lx = img0.shape[-2:]
    ysub = np.arange(xpad1, xpad1 + Ly)
    xsub = np.arange(ypad1, ypad1 + Lx)
    return I, ysub, xsub


def random_rotate_and_resize(X, Y=None, scale_range=1., xy=(224, 224),
                             do_flip=True, rescale=None, unet=False):
    """ Augmentation of input images

    This function handles augmentation of input images by random rotation and resizing.

    Args:
        X(list[float]): List of image arrays of size[Ly x Lx]
        Y(list[float]): List of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
        of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
        If Y.shape[0]==3 . The labels are assumed to be [cell probability, Y flow, X flow].
        scale_range(float): Default value 1.0. Range of resizing of images for augmentation. Images are resized by
        (1-scale_range/2) + scale_range * np.random.rand()
        xy(tuple[int]): Default value (224,224). Size of transformed images to return
        do_flip(bool): Default True. Whether or not to flip images horizontally
        rescale(array[float]): Default None. How much to resize images by before performing augmentations
        unet(bool): Set true to use Unet model
    Returns:
        imgi(array[float]): Transformed images in array [nimg x nchan x xy[0] x xy[1]]
        lbl(array[float]): Transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        scale(array[float]): Amount each image was resized by

    """

    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim > 2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)
    lbl = []
    if Y is not None:
        if Y[0].ndim > 2:
            nt = Y[0].shape[0]
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)
    scale = np.zeros(nimg, np.float32)
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]
        # generate random augmentation parameters
        flip = np.random.rand() > .5
        theta = np.random.rand() * np.pi * 2
        scale[n] = (1 - scale_range / 2) + scale_range * np.random.rand()
        if rescale is not None:
            scale[n] *= 1. / rescale[n]
        dxy = np.maximum(0, np.array([Lx * scale[n] - xy[1], Ly * scale[n] - xy[0]]))
        dxy = (np.random.rand(2, ) - .5) * dxy
        # create affine transform
        cc = np.array([Lx / 2, Ly / 2])
        cc1 = cc - np.array([Lx - xy[1], Ly - xy[0]]) / 2 + dxy
        pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
        pts2 = np.float32([cc1,
                           cc1 + scale[n] * np.array([np.prediction.cos(theta), np.sin(theta)]),
                           cc1 + scale[n] * np.array(
                               [np.cos(np.pi / 2 + theta), np.sin(np.pi / 2 + theta)])])
        M = cv2.getAffineTransform(pts1, pts2)
        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim < 3:
                labels = labels[np.newaxis, :, :]
        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1 and not unet:
                    labels[2] = -labels[2]
        for k in range(nchan):
            I = cv2.warpAffine(img[k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)
            imgi[n, k] = I
        if Y is not None:
            for k in range(nt):
                if k == 0:
                    lbl[n, k] = cv2.warpAffine(labels[k], M, (xy[1], xy[0]),
                                               flags=cv2.INTER_NEAREST)
                else:
                    lbl[n, k] = cv2.warpAffine(labels[k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)
            if nt > 1 and not unet:
                v1 = lbl[n, 2].copy()
                v2 = lbl[n, 1].copy()
                lbl[n, 1] = (-v1 * np.sin(-theta) + v2 * np.cos(-theta))
                lbl[n, 2] = (v1 * np.cos(-theta) + v2 * np.sin(-theta))
    return imgi, lbl, scale


def _X2zoom(img, X2=1):
    """ Zoom in image

    This function zooms into a image.

    Args:
        img(array): Numpy array that's Ly x Lx

    Returns:
        img(array): Numpy array that's Ly x Lx

    """

    ny, nx = img.shape[:2]
    img = cv2.resize(img, (int(nx * (2 ** X2)), int(ny * (2 ** X2))))
    return img


def _image_resizer(img, resize=512, to_uint8=False):
    """ Resize image

    Function to resize image.

    Args:
        img(array): Numpy array that's Ly x Lx
        resize(int): Max size of image returned
        to_uint8(bool): Convert image to uint8
    Returns:
        img(array): Numpy array that's Ly x Lx

    """

    ny, nx = img.shape[:2]
    if to_uint8:
        if img.max() <= 255 and img.min() >= 0 and img.max() > 1:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)
    if np.array(img.shape).max() > resize:
        if ny > nx:
            nx = int(nx / ny * resize)
            ny = resize
        else:
            ny = int(ny / nx * resize)
            nx = resize
        shape = (nx, ny)
        img = cv2.resize(img, shape)
        img = img.astype(np.uint8)
    return img
