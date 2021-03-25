import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile

from scipy.ndimage import median_filter
import cv2
import transforms, dynamics, utils, metrics, core
from core import UnetModel, assign_device, check_mkl, use_gpu, convert_images, parse_model_string

urls = [
        'https://www.cellpose.org/models/cytotorch_0',
        'https://www.cellpose.org/models/cytotorch_1',
        'https://www.cellpose.org/models/cytotorch_2',
        'https://www.cellpose.org/models/cytotorch_3',
        'https://www.cellpose.org/models/size_cytotorch_0.npy',
        'https://www.cellpose.org/models/nucleitorch_0',
        'https://www.cellpose.org/models/nucleitorch_1',
        'https://www.cellpose.org/models/nucleitorch_2',
        'https://www.cellpose.org/models/nucleitorch_3',
        'https://www.cellpose.org/models/size_nucleitorch_0.npy']


def download_model_weights(urls=urls):
    # cellpose directory
    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    model_dir = cp_dir.joinpath('models')
    model_dir.mkdir(exist_ok=True)

    for url in urls:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            utils.download_url_to_file(url, cached_file, progress=True)


download_model_weights()
model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')


def dx_to_circ(dP):
    """ dP is 2 x Y x X => 'optic' flow representation """
    sc = max(np.percentile(dP[0], 99), np.percentile(dP[0], 1))
    Y = np.clip(dP[0] / sc, -1, 1)
    sc = max(np.percentile(dP[1], 99), np.percentile(dP[1], 1))
    X = np.clip(dP[1] / sc, -1, 1)
    H = (np.arctan2(Y, X) + np.pi) / (2 * np.pi)
    S = utils.normalize99(dP[0] ** 2 + dP[1] ** 2)
    V = np.ones_like(S)
    HSV = np.concatenate((H[:, :, np.newaxis], S[:, :, np.newaxis], S[:, :, np.newaxis]), axis=-1)
    HSV = np.clip(HSV, 0.0, 1.0)
    flow = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return flow


class CellposeModel(UnetModel):
    """

    Parameters
    -------------------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available

    pretrained_model: str or list of strings (optional, default False)
        path to pretrained cellpose model(s), if False, no model loaded;
        if None, built-in 'cyto' model loaded

    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False

    diam_mean: float (optional, default 27.)
        mean 'diameter', 27. is built in value for 'cyto' model

    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))

    """

    def __init__(self, gpu=False, pretrained_model=False, torch=True,
                 diam_mean=30., net_avg=True, device=None,
                 residual_on=True, style_on=True, concatenation=False):
        torch = True
        self.torch = torch
        sdevice, gpu = assign_device()
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        nclasses = 3  # 3 prediction maps (dY, dX and cellprob)
        self.nclasses = nclasses
        if pretrained_model:
            params = parse_model_string(pretrained_model)
            if params is not None:
                nclasses, residual_on, style_on, concatenation = params
        # load default cyto model if pretrained_model is None
        elif pretrained_model is None:
            torch_str = ['', 'torch'][self.torch]
            pretrained_model = [os.fspath(model_dir.joinpath('cyto%s_%d' % (torch_str, j))) for j in
                                range(4)] if net_avg else os.fspath(model_dir.joinpath('cyto_0'))
            self.diam_mean = 30.
            residual_on, style_on, concatenation = True, True, False

        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=diam_mean, net_avg=net_avg, device=sdevice,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                         nclasses=nclasses, torch=torch)
        self.unet = False
        self.pretrained_model = pretrained_model
        if self.pretrained_model is not None and isinstance(self.pretrained_model, str):
            self.net.load_model(self.pretrained_model, cpu=(not self.gpu))
        ostr = ['off', 'on']
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])


    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """

        veci = 5. * self._to_device(lbl[:, 1:])
        lbl = self._to_device(lbl[:, 0] > .5)
        loss = self.criterion(y[:, :2], veci)
        if self.torch:
            loss /= 2.
        loss2 = self.criterion2(y[:, -1], lbl)
        loss = loss + loss2
        return loss

    def train(self, train_data, train_labels, train_files=None,
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None,
              save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, batch_size=8, rescale=True):

        """ train network with images train_data

            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            train_files: list of strings
                file names for images in train_data (to save flows for future runs)

            test_data: list of arrays (2D or 3D)
                images for testing

            test_labels: list of arrays (2D or 3D)
                labels for test_data, where 0=no masks; 1,2,...=mask labels;
                can include flows as additional images

            test_files: list of strings
                file names for images in test_data (to save flows for future runs)

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            pretrained_model: string (default, None)
                path to pretrained_model to start from, if None it is trained from scratch

            save_path: string (default, None)
                where to save trained model, if None it is not saved

            save_every: int (default, 100)
                save network every [save_every] epochs

            learning_rate: float (default, 0.2)
                learning rate for training

            n_epochs: int (default, 500)
                how many times to go through whole training set during training

            weight_decay: float (default, 0.00001)

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            rescale: bool (default, True)
                whether or not to rescale images to diam_mean during training,
                if True it assumes you will fit a size model after training or resize your images accordingly,
                if False it will try to train the model to be scale-invariant (works worse)

        """

        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data,
                                                                                                   train_labels ,
                                                                                                   test_data,
                                                                                                   test_labels,
                                                                                                   channels, normalize)

        # check if train_labels have flows
      #  train_flows = dynamics.labels_to_flows(train_labels, files=train_files)
      #   train_flows =[]
      #   train_flows - train_labels
        # if run_test:
        #     test_flows = dynamics.labels_to_flows(test_labels, files=test_files)
        # else:
        #     test_flows = None

        model_path = self._train_net(train_data, train_labels,
                                     test_data, test_labels,
                                     pretrained_model, save_path, save_every,
                                     learning_rate, n_epochs, momentum, weight_decay, batch_size, rescale)

        self.pretrained_model = model_path
        return model_path
