import warnings

import cv2
import numpy as np


def normalize99(img):
    """ Normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile
    Args:
        img(array[float]): Image
    Returns:
        img(array[float]): Normalised image

    """
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X


def reshape(data, channels=[0, 0], chan_first=False):
    """ Reshape data using channels
    Args:
    data(array) : Numpy array that's  Ly x Lx x nchan
        if data.ndim==8 and data.shape[0]<8, assumed to be nchan x Ly x Lx
    channels(list[int]): channel to segment
    invert(bool) : Invert intensities
    Returns:
    data(array): Numpy array that's (Z x ) Ly x Lx x nchan (if chan_first==False)

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
                        warnings.warn("chan to seg' has value range of ZERO")
                    else:
                        warnings.warn(
                            "'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
            if data.shape[-1] == 1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim == 4:
            data = np.transpose(data, (3, 0, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))
    return data


def normalize_img(img, axis=-1, invert=False):
    """ Normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities.
    Args:
    img(array[float]): ND-array. Unlabeled array
    axis(int): Channel axis to loop over for normalization
    Returns:
    img(array[float]): Normalized image of same size

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


def reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize):
    """ Check sizes and reshape train and test data for training
    Args:
    train_data(list[float]): List of training images of size [Ly x Lx]
    train_labels(list[float]): List of training labels of size [Ly x Lx x 3]
    test_data(list[float]): List of testing images of size [Ly x Lx]
    test_labels(list[float]): List of testing labels of size [Ly x Lx x 3]
    channels(list[int]): Channel to segment
    normalize(bool): Normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
    Returns:
    train_data(list[float]): List of training images of size [2 x Ly x Lx]
    train_labels(list[float]): List of training labels of size [Ly x Lx x 3]
    test_data(list[float]): List of testing images of size [2 x Ly x Lx]
    test_labels(list[float]): List of testing labels of size [Ly x Lx x 3]
    run_test(bool): Whether or not test_data was correct size and is usable during training

    """
    nimg = len(train_data)
    # check that arrays are correct size
    if nimg != len(train_labels):
        raise ValueError('train data and labels not same length')
        return
    if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
        raise ValueError('training data or labels are not at least two-dimensional')
        return

    if train_data[0].ndim > 3:
        raise ValueError('training data is more than three-dimensional (should be 2D or 3D array)')
        return

    # check if test_data correct length
    if not (test_data is not None and test_labels is not None and
            len(test_data) > 0 and len(test_data) == len(test_labels)):
        test_data = None

    # make data correct shape and normalize it so that 0 and 1 are 1st and 99th percentile of data
    train_data, test_data, run_test = reshape_and_normalize_data(train_data, test_data=test_data,
                                                                 channels=channels,
                                                                 normalize=normalize)

    if train_data is None:
        raise ValueError('training data do not all have the same number of channels')
        return

    if not run_test:
        print(
            'NOTE: test data not provided OR labels incorrect OR not same number of channels as train data')
        test_data, test_labels = None, None

    return train_data, train_labels, test_data, test_labels, run_test


def reshape_and_normalize_data(train_data, test_data=None, channels=None, normalize=True):
    """ Inputs converted to correct shapes for *training* and rescaled so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities in each channel
    Args:
    train_data(list[float]): List of training images of size [Ly x Lx]
    test_data(list[float]): List of testing images of size [Ly x Lx]
    channels(list[int]): channel to segment
    normalize(bool): Normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
    Returns:
    train_data(list[float]): List of training images of size [2 x Ly x Lx]
    test_data(list[float]): List of testing images of size [2 x Ly x Lx]
    run_test(bool): Whether or not test_data was correct size and is usable during training

    """
    # if training data is less than 2D
    nimg = len(train_data)
    if channels is not None:
        train_data = [reshape(train_data[n], channels=channels, chan_first=True) for n in
                      range(nimg)]
    if train_data[0].ndim < 3:
        train_data = [train_data[n][:, :, np.newaxis] for n in range(nimg)]
    elif train_data[0].shape[-1] < 8:
        print(
            'NOTE: assuming train_data provided as Ly x Lx x nchannels, transposing axes to put channels first')
        train_data = [np.transpose(train_data[n], (2, 0, 1)) for n in range(nimg)]
    nchan = [train_data[n].shape[0] for n in range(nimg)]
    if nchan.count(nchan[0]) != len(nchan):
        return None, None, None
    nchan = nchan[0]

    # check for valid test data
    run_test = False
    if test_data is not None:
        nimgt = len(test_data)
        if channels is not None:
            test_data = [reshape(test_data[n], channels=channels, chan_first=True) for n in
                         range(nimgt)]
        if test_data[0].ndim == 2:
            if nchan == 1:
                run_test = True
                test_data = [test_data[n][np.newaxis, :, :] for n in range(nimgt)]
        elif test_data[0].ndim == 3:
            if test_data[0].shape[-1] < 8:
                print(
                    'NOTE: assuming test_data provided as Ly x Lx x nchannels, transposing axes to put channels first')
                test_data = [np.transpose(test_data[n], (2, 0, 1)) for n in range(nimgt)]
            nchan_test = [test_data[n].shape[0] for n in range(nimgt)]
            if nchan_test.count(nchan_test[0]) != len(nchan_test):
                run_test = False
            elif test_data[0].shape[0] == nchan:
                run_test = True

    if normalize:
        train_data = [normalize_img(train_data[n], axis=0) for n in range(nimg)]
        if run_test:
            test_data = [normalize_img(test_data[n], axis=0) for n in range(nimgt)]

    return train_data, test_data, run_test


def random_rotate_and_resize(X, Y=None, scale_range=1., xy=(224, 224),
                             do_flip=True, rescale=None, unet=False):
    """ Augmentation by random rotation and resizing.X and Y are lists or arrays of length nimg.
    Args:
        X(list[float]): List of image arrays of size [Ly x Lx]
        Y(list[float]): List of image labels of size [nlabels x Ly x Lx] or [Ly x Lx].If Y.shape[0]==3
                        then the labels are assumed to be [cell probability, Y flow, X flow].
        scale_range(float): Range of resize of images for augmentation. Images are resized by
        (1-scale_range/2) + scale_range * np.random.rand()
        xy(tuple[int]): Size of transformed images to return
        do_flip(bool): Whether or not to flip images horizontally
        rescale(array[float]): How much to resize images by before performing augmentations
        unet(bool): Set if unet model is used
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
                           cc1 + scale[n] * np.array([np.cos(theta), np.sin(theta)]),
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
