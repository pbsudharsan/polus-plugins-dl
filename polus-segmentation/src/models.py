# Code sourced from https://github.com/MouseLand/cellpose/tree/master/cellpose

import os
import pathlib

import numpy as np

import transforms
from core import UnetModel, assign_device, parse_model_string

model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')


class CellposeModel(UnetModel):
    """ Class for cellpose model
    Args:
     gpu(bool): Whether or not to save model to GPU, will check if GPU available
     pretrained_model(str): Path to pretrained cellpose model(s).
     net_avg(bool): Default True. Loads the 4 built-in networks and averages them if True, loads one network if False
     diam_mean(float): Default 27. Mean 'diameter', 27. Is built in value for 'cyto' model
     device(torch): Where model is saved (torch.gpu() or torch.cpu())

    """

    def __init__(self, gpu=False, model_type=None, pretrained_model=False, torch=True,
                 diam_mean=30., net_avg=True, device=None,
                 residual_on=True, style_on=True, concatenation=False):
        torch = True
        self.torch = torch
        sdevice, gpu = assign_device()
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        elif isinstance(pretrained_model, str):
            pretrained_model = [pretrained_model]
        nclasses = 3  # 3 prediction maps (dY, dX and cellprob)
        self.nclasses = nclasses

        # load default cyto model if pretrained_model is None
        if model_type in ['cyto', 'nuclei']:
            torch_str = ['', 'torch'][self.torch]
            pretrained_model = [os.fspath(model_dir.joinpath(
                '%s%s_%d' % (pretrained_model, torch_str, j)))
                for j in range(4)]
            pretrained_model = pretrained_model[0] if not net_avg else pretrained_model
            diam_mean = 30. if pretrained_model == 'cyto' else 17.
            residual_on, style_on, concatenation = True, True, False

        else:
            if pretrained_model:
                params = parse_model_string(pretrained_model)
                if params is not None:
                    nclasses, residual_on, style_on, concatenation = params
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
        """ Loss function between true labels lbl and prediction y
        Args:
            lbl(array[float32]): Labeled images
            y(array[float32]) : Predicted label of given images
        Returns:
            loss(float): BCEWithLogitsLoss + MSE loss

        """
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
              save_path=None, save_every=20,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, batch_size=8,
              rescale=True):
        """ Train network with images train_data
        Args:
            train_data(list[array]): Images for training
            train_labels(list[array]): Labels for train_data, where 0=no masks; 1,2,...=mask labels
            train_files(list[string]): File names for images in train_data (to save flows for future runs)
            test_data(list[array]): Images for testing
            test_labels(list[array]): Labels for test_data, where 0=no masks; 1,2,...=mask labels
            test_files(list[string]): File names for images in test_data (to save flows for future runs
            channels(list[int]): Channels to use for training
            normalize(bool): Normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
            pretrained_model(string): Path to pretrained_model to start from, if None it is trained from scratch
            save_path(string): Where to save trained model, if None it is not saved
            save_every(int): Default 100. Save network every [save_every] epochs
            learning_rate(float): Default 0.2. Learning rate for training
            n_epochs(int): Default 500. How many times to go through whole training set during training
            weight_decay(float): Default 0.00001. Weight decay
            batch_size(int): Default 8. Number of 224x224 patches to run simultaneously on the GPU
            rescale(bool): Whether or not to rescale images to diam_mean during training
        Returns:
            model_path(str): Model path

        """
        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(
            train_data,
            train_labels,
            test_data,
            test_labels,
            channels, normalize)

        model_path = self._train_net(train_data, train_labels,
                                     test_data, test_labels,
                                     pretrained_model, save_path, save_every,
                                     learning_rate, n_epochs, momentum, weight_decay, batch_size,
                                     rescale)

        self.pretrained_model = model_path
        return model_path
