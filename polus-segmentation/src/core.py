# Code sourced from https://github.com/MouseLand/cellpose/tree/master/cellpose

import datetime
import logging
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from torch import optim, nn

import resnet_torch
import transforms
import utils

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("arch")
logger.setLevel(logging.INFO)
TORCH_ENABLED = True
torch_GPU = torch.device('cuda')
torch_CPU = torch.device('cpu')


def parse_model_string(pretrained_model):
    """ Split the input pretrained model name
    Args:
        pretrained_model(string): Pretrained model input
    Returns:
        nclasses(int): Number of classes to segment
        residual_on(bool): Set residual connections in network
        style_on(bool): Set styles for network
        concatenation(bool): Set connection between upsampling and downsampling in network

    """
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str) > 3 and model_str[:4] == 'unet':
        logger.info('parsing model string to get unet options')
        nclasses = max(2, int(model_str[4]))
    elif len(model_str) > 7 and model_str[:8] == 'cellpose':
        logger.info('parsing model string to get cellpose options')
        nclasses = 3
    else:
        return None
    ostrs = model_str.split('_')[2::2]
    residual_on = ostrs[0] == 'on'
    style_on = ostrs[1] == 'on'
    concatenation = ostrs[2] == 'on'
    return nclasses, residual_on, style_on, concatenation


def use_gpu(gpu_number=0):
    """ Check if gpu works
    Args:
        gpu_number(int): Gpu number
        istorch(bool): Use of torch
    Returns:
        _(bool) : True if gpu is present else false
    """
    return _use_gpu_torch(gpu_number)


def _use_gpu_torch(gpu_number=0):
    """ Checks for Cuda installation
    Args:
        gpu_number(int): Gpu number
    Returns:
        _(bool): True or False
    """
    try:
        device = torch.device('cuda:' + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        logger.info('TORCH CUDA version installed and working.')
        return True
    except:
        logger.info('TORCH CUDA version not installed/working.')
        return False


def assign_device():
    """ Setting CUDA/CPU
    Returns:
        device(torch.device): Set cpu/gpu
        pu(bool): True if gpu is being used
    """
    if _use_gpu_torch():
        device = torch_GPU
        gpu = True
        logger.info('Using GPU')
    else:
        device = torch_CPU
        logger.info('Using CPU')
        gpu = False
    return device, gpu


def check_mkl(istorch=True):
    """ Test snippet to check if MKL-DNN working
    Args:
        istorch(bool): Checks for torch usage
    Returns:
        mkl_enabled(bool): True if cpu is mkl enabled

    """
    print('Running test snippet to check if MKL-DNN working')
    if istorch:
        logger.info('See https://pytorch.org/docs/stable/backends.html?highlight=mkl')
        mkl_enabled = torch.backends.mkldnn.is_available()

    if mkl_enabled:
        logger.info('MKL version working - CPU version is sped up.')

    else:
        logger.info(
            'WARNING: MKL version on torch not working/installed - CPU version will be slightly slower.')
    return mkl_enabled


class UnetModel():
    """ Class for Unet model
    Args:
         pretrained_model(string): Pretrained model input
         net_avg(bool): Default True.Loads the 4 built-in networks and averages them if True, loads one network if False
         diam_mean(float): Default 27.Mean 'diameter', 27. is built in value for 'cyto' model
         device(torch): Where model is saved (torch.gpu() or torch.cpu()), overrides gpu input,
                    recommended if you want to use a specific GPU (e.g. mx.gpu(4))
         residual_on(bool): Set residual connections in network
         style_on(bool): Set styles for network
         concatenation(bool): Set connection between upsampling and downsampling in network

    """

    def __init__(self, gpu=True, pretrained_model=False,
                 diam_mean=30., net_avg=True, device=None,
                 residual_on=False, style_on=False, concatenation=True,
                 nclasses=3, torch=True):
        self.unet = True

        torch = True
        self.torch = torch
        self.mkldnn = None
        self.device = device
        self.gpu = gpu
        if torch and not self.gpu:
            self.mkldnn = check_mkl(self.torch)
        self.pretrained_model = pretrained_model
        self.diam_mean = diam_mean

        if pretrained_model:
            params = parse_model_string(pretrained_model)
            if params is not None:
                nclasses, residual_on, style_on, concatenation = params

        ostr = ['off', 'on']
        self.net_type = 'unet{}_residual_{}_style_{}_concatenation_{}'.format(nclasses,
                                                                              ostr[residual_on],
                                                                              ostr[style_on],
                                                                              ostr[concatenation])
        if pretrained_model:
            print(self.net_type)
        # create network
        self.nclasses = nclasses
        nbase = [32, 64, 128, 256]
        if self.torch:
            nchan = 2
            nbase = [nchan, 32, 64, 128, 256]
            self.net = resnet_torch.CPnet(nbase,
                                          self.nclasses,
                                          3,
                                          residual_on=residual_on,
                                          style_on=style_on,
                                          concatenation=concatenation,
                                          mkldnn=self.mkldnn).to(self.device)

        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_model(pretrained_model, cpu=(not self.gpu))

    def _to_device(self, x):
        """ Convert numpy to tensor
        Args:
            x(array): Numpy array
        Returns:
            X(tensor): Tensor array

        """
        if self.torch:
            X = torch.from_numpy(x).float().to(self.device)

        return X

    def _from_device(self, X):
        """ Convert tensor to numpy
        Args:
            X(tensor): Tensor array
        Returns:
            x(array): Numpy array

        """
        if self.torch:
            x = X.detach().cpu().numpy()

        return x

    def loss_fn(self, lbl, y):
        """ Loss function between true labels lbl and prediction y
        Args:
            lbl(array[float32]): Labelled array
            y(array[float32]): Predicted Label array
        Returns:
            loss(float): Loss between predicted and labelled array

        """
        # if available set boundary pixels to 2
        if lbl.shape[1] > 1 and self.nclasses > 2:
            boundary = lbl[:, 1] <= 4
            lbl = lbl[:, 0]
            lbl[boundary] *= 2
        else:
            lbl = lbl[:, 0]
        lbl = self._to_device(lbl)
        loss = 8 * 1. / self.nclasses * self.criterion(y, lbl)
        return loss

    def _train_step(self, x, lbl):
        """  Function to calculate train loss
        Args:
            x(array[float32]): Unlabelled array
            lbl(array[float32]): Labelled array
        Returns:
            train_loss(float): Train loss

        """

        X = self._to_device(x)
        if self.torch:
            self.optimizer.zero_grad()
            if self.gpu:
                self.net.train().cuda()
            else:
                self.net.train()
            y, style = self.net(X)
            loss = self.loss_fn(lbl, y)

            loss.backward()
            train_loss = loss.item()
            self.optimizer.step()
            train_loss *= len(x)

        return train_loss

    def _test_eval(self, x, lbl):
        """ Test evaluation
        Args:
            x(array[float32]): Unlabelled array
            lbl(array[float32]): Labelled array
        Returns:
            test_loss(float): Test loss

        """
        X = self._to_device(x)
        if self.torch:
            self.net.eval()
            y, style = self.net(X)
            loss = self.loss_fn(lbl, y)
            test_loss = loss.item()
            test_loss *= len(x)

        return test_loss

    def _set_optimizer(self, learning_rate, momentum, weight_decay):
        """ Optimizing function
        Args:
            learning_rate(float): Learning rate
            momentum(float): Momentum. Optimization parameter
            weight_decay(float): Weight decay
        """
        if self.torch:
            self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate,
                                       momentum=momentum, weight_decay=weight_decay)

    def _set_learning_rate(self, lr):
        """ Set learning rate
        Args:
            learning_rate(float): Learning rate

        """
        if self.torch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def _set_criterion(self):
        """ Set loss criterion

        """
        if self.torch:
            self.criterion = nn.MSELoss(reduction='mean')
            self.criterion2 = nn.BCEWithLogitsLoss(reduction='mean')

    def _train_net(self, train_data, train_labels,
                   test_data=None, test_labels=None,
                   pretrained_model=None, save_path=None, save_every=100,
                   learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001,
                   batch_size=8, rescale=True, netstr='cellpose', early_stopping=5):
        """ Train function uses loss function self.loss_fn
        Args:
            train_data(list[array]): Images for training
            train_labels(list[array]): Labels for train_data, where 0=no masks; 1,2,...=mask labels
            test_data(list[array]): Images for testing
            test_labels(list[array]): Labels for test_data, where 0=no masks; 1,2,...=mask labels
            pretrained_model(string): Path to pretrained_model to start from, if None it is trained from scratch
            save_path(string): Where to save trained model, if None it is not saved
            save_every(int): Default 100. Save network every [save_every] epochs
            learning_rate(float): Default 0.2. Learning rate for training
            n_epochs(int): Default 500. How many times to go through whole training set during training
            weight_decay(float): Default 0.00001. Weight decay
            batch_size(int): Default 8. Number of 224x224 patches to run simultaneously on the GPU
            rescale(bool): Whether or not to rescale images to diam_mean during training
            early_stopping(int): Stop training if there is no imporvement in test loss
        Returns:
            model_path(str): Model path

         """
        d = datetime.datetime.now()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self._set_optimizer(self.learning_rate, momentum, weight_decay)

        self._set_criterion()

        nimg = len(train_data)

        # compute average cell diameter
        if rescale:
            diam_train = np.array(
                [utils.diameters(train_labels[k][0])[0] for k in range(len(train_labels))])
            diam_train[diam_train < 5] = 5.
            if test_data is not None:
                diam_test = np.array(
                    [utils.diameters(test_labels[k][0])[0] for k in range(len(test_labels))])
                diam_test[diam_test < 5] = 5.
            scale_range = 0.5
        else:
            scale_range = 1.0
        loss_history = deque(maxlen=early_stopping + 1)
        nchan = train_data[0].shape[0]
        logger.info('Training network with %d channel input ' % nchan)
        logger.info('Saving every %d epochs' % save_every)
        logger.info('Median diameter = %d' % self.diam_mean)
        logger.info(
            'LR: %0.5f, batch_size: %d, weight_decay: %0.5f' % (
            self.learning_rate, self.batch_size, weight_decay))
        logger.info('Ntrain = %d' % nimg)
        if test_data is not None:
            logger.info('Ntest = %d' % len(test_data))

        # set learning rate schedule
        LR = np.linspace(0, self.learning_rate, 10)
        if self.n_epochs > 250:
            LR = np.append(LR, self.learning_rate * np.ones(self.n_epochs - 100))
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10))
        else:
            LR = np.append(LR, self.learning_rate * np.ones(max(0, self.n_epochs - 10)))

        tic = time.time()

        lavg, nsum = 0, 0

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path)

            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            logger.info('WARNING: no save_path given, model not saving')

        ksave = 0
        rsc = 1.0

        # cannot train with mkldnn
        self.net.mkldnn = False

        for iepoch in range(self.n_epochs):
            np.random.seed(iepoch)
            rperm = np.random.permutation(nimg)
            self._set_learning_rate(LR[iepoch])

            for ibatch in range(0, nimg, batch_size):
                inds = rperm[ibatch:ibatch + batch_size]
                rsc = diam_train[inds] / self.diam_mean if rescale else np.ones(len(inds),
                                                                                np.float32)
                imgi, lbl, scale = transforms.random_rotate_and_resize(
                    [train_data[i] for i in inds], Y=[train_labels[i][1:] for i in inds],
                    rescale=rsc, scale_range=scale_range, unet=self.unet)

                train_loss = self._train_step(imgi, lbl)
                lavg += train_loss
                nsum += len(imgi)

            if iepoch % 10 == 0 or iepoch < 10:
                lavg = lavg / nsum
                if test_data is not None:
                    lavgt, nsum = 0., 0
                    np.random.seed(42)
                    rperm = np.arange(0, len(test_data), 1, int)
                    for ibatch in range(0, len(test_data), batch_size):
                        inds = rperm[ibatch:ibatch + batch_size]
                        rsc = diam_test[inds] / self.diam_mean if rescale else np.ones(len(inds),
                                                                                       np.float32)
                        imgi, lbl, scale = transforms.random_rotate_and_resize(
                            [test_data[i] for i in inds],
                            Y=[test_labels[i][1:] for i in inds],
                            scale_range=0., rescale=rsc, unet=self.unet)
                        if self.unet and lbl.shape[1] > 1 and rescale:
                            lbl[:, 1] *= scale[0] ** 2

                        test_loss = self._test_eval(imgi, lbl)

                        lavgt += test_loss
                        nsum += len(imgi)
                    loss_history.append(lavgt / nsum)
                    logger.info('Epoch %d, Time %4.1fs, Loss %2.4f, Loss Test %2.4f, LR %2.4f' %
                                (iepoch, time.time() - tic, lavg, lavgt / nsum, LR[iepoch]))
                    if len(loss_history) > early_stopping:
                        if loss_history.popleft() < min(loss_history):
                            logger.info(f'\nEarly stopping. No Test loss '
                                        f'improvement in {early_stopping} epochs.')
                            file = '{}_{}_{}'.format(self.net_type, file_label,
                                                     d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                            logger.info('Saving network parameters')
                            self.net.save_model(os.path.join(file_path, file))
                            sys.exit()

                else:
                    logger.info('Epoch %d, Time %4.1fs, Loss %2.4f, LR %2.4f' %
                                (iepoch, time.time() - tic, lavg, LR[iepoch]))
                lavg, nsum = 0, 0

            # if save_path is not None:
            if iepoch == self.n_epochs - 1 or iepoch % save_every == 1:
                # save model at the end
                file = '{}_{}_{}'.format(self.net_type, file_label,
                                         d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                ksave += 1
                logger.info('Saving network parameters')
                self.net.save_model(os.path.join(file_path, file))

        # reset to mkldnn if available
        self.net.mkldnn = self.mkldnn

        return file
