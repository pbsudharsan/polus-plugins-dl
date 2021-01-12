import os, logging
import numpy as np
import transforms, dynamics, utils, metrics
import torch
from torch import optim, nn
from torch.utils import mkldnn as mkldnn_utils
import resnet_torch

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("arch")
logger.setLevel(logging.INFO)
TORCH_ENABLED = True
torch_GPU = torch.device('cuda')
torch_CPU = torch.device('cpu')


def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str)>3 and model_str[:4]=='unet':
        print('parsing model string to get unet options')
        nclasses = max(2, int(model_str[4]))
    elif len(model_str)>7 and model_str[:8]=='cellpose':
        print('parsing model string to get cellpose options')
        nclasses = 3
    else:
        return None
    ostrs = model_str.split('_')[2::2]
    residual_on = ostrs[0]=='on'
    style_on = ostrs[1]=='on'
    concatenation = ostrs[2]=='on'
    return nclasses, residual_on, style_on, concatenation

def use_gpu(gpu_number=0, istorch=True):
    """ check if gpu works """
    if istorch:
        return _use_gpu_torch(gpu_number)

def _use_gpu_torch(gpu_number=0):
    try:
        device = torch.device('cuda:' + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        logger.info(' TORCH CUDA version installed and working. ')
        return True
    except:
        logger.info('TORCH CUDA version not installed/working.')
        return False


def assign_device(istorch):
    if  use_gpu(istorch=istorch):
        device = torch_GPU
        gpu=True
        logger.info(' using GPU')
    else:
        device = torch_CPU
        logger.info(' using CPU')
        gpu=False
    return device, gpu

def check_mkl(istorch=True):
    logger.info('Running test snippet to check if MKL-DNN working')
    if istorch:
        logger.info('see https://pytorch.org/docs/stable/backends.html?highlight=mkl')
        mkl_enabled = torch.backends.mkldnn.is_available()

    if mkl_enabled:
        logger.info(' MKL version working - CPU version is sped up. ')
    else:
        logger.info('WARNING: MKL version on torch not working/installed - CPU version will be slightly slower.')
    return mkl_enabled


def convert_images(x, channels, normalize, invert):
    """ return list of images with channels last and normalized intensities """
    if not isinstance(x,list) and not (x.ndim>3 ):
        nolist = True
        x = [x]
    else:
        nolist = False
    
    nimg = len(x)

    if channels is not None:
        if len(channels)==2:
            if not isinstance(channels[0], list):
                channels = [channels for i in range(nimg)]
        for i in range(len(x)):
            if x[i].shape[0]<4:
                x[i] = x[i].transpose(1,2,0)
        x = [transforms.reshape(x[i], channels=channels[i]) for i in range(nimg)]

    else:
        for i in range(len(x)):
            if x[i].ndim>3:
                raise ValueError('ERROR: cannot process 4D images ')
            elif x[i].ndim==2:
                x[i] = np.stack((x[i], np.zeros_like(x[i])), axis=2)
            elif x[i].shape[0]<8:
                x[i] = x[i].transpose((1,2,0))
            if x[i].shape[-1]>2:
                print('WARNING: more than 2 channels given, use "channels" input for specifying channels - just using first two channels to run processing')
                x[i] = x[i][:,:,:2]

    if normalize or invert:
        x = [transforms.normalize_img(x[i], invert=invert) for i in range(nimg)]
    return x, nolist


class UnetModel():
    def __init__(self, pretrained_model=False,gpu=False,
                    diam_mean=30., net_avg=True, device=None,
                    residual_on=False, style_on=False, concatenation=True,
                    nclasses = 3, torch=True):
        self.unet = True
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
        nbase = [32,64,128,256]
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
        X = torch.from_numpy(x).float().to(self.device)
        return X

    def _from_device(self, X):
        x = X.detach().cpu().numpy()
        return x

    def network(self, x):
        """ convert imgs to torch and run network model and return numpy """
        X = self._to_device(x)
        if self.torch:
            self.net.eval()
            if self.mkldnn:
                self.net = mkldnn_utils.to_mkldnn(self.net)
        y, style = self.net(X)
        if self.mkldnn:
            self.net.to(torch_CPU)
        y = self._from_device(y)
        style = self._from_device(style)
        return y,style

    def _run_nets(self, img, net_avg=True, augment=False, tile=True, tile_overlap=0.1, bsize=224, progress=None):
        """ run network (if more than one, loop over networks and average results
        Args:
        img(array[float]):  [Ly x Lx x nchan]
        net_avg(bool):  default True.runs the 4 built-in networks and averages them if True, runs one network if False
        augment(bool): default False.tiles image with overlapping tiles and flips overlapped regions to augment
        tile(bool): default True.tiles image to ensure GPU memory usage limited (recommended)
        tile_overlap(float):  default 0.1.fraction of overlap of tiles when computing flows

        Returns:
        y(array):  [3 x Ly x Lx] y is output (averaged over networks);
            y[0] is Y flow; y[1] is X flow; y[2] is cell probability
        style(array): 1D array summarizing the style of the image,if tiled it is averaged over tiles,
            but not averaged over networks.

        """
        if isinstance(self.pretrained_model, str) or not net_avg:
            y, style = self._run_net(img, augment=augment, tile=tile, bsize=bsize)
        else:
            for j in range(len(self.pretrained_model)):
                self.net.load_model(self.pretrained_model[j], cpu=(not self.gpu))
                if not self.torch:
                    self.net.collect_params().grad_req = 'null'
                y0, style = self._run_net(img, augment=augment, tile=tile,
                                          tile_overlap=tile_overlap, bsize=bsize)

                if j==0:
                    y = y0
                else:
                    y += y0
                if progress is not None:
                    progress.setValue(10 + 10*j)
            y = y / len(self.pretrained_model)
        return y, style

    def _run_net(self, imgs, augment=False, tile=True, tile_overlap=0.1, bsize=224):
        """ run network on image or stack of images(faster if augment is False)
        Args:
        imgs(array):  [Ly x Lx x nchan]
        rsz(float):  default 1.0. resize coefficient(s) for image
        augment(bool): default False.tiles image with overlapping tiles and flips overlapped regions to augment
        tile(bool): default True.tiles image to ensure GPU/CPU memory usage limited (recommended);
        tile_overlap(float): default 0.1.fraction of overlap of tiles when computing flows
        bsize(int):  default 224.size of tiles to use in pixels [bsize x bsize]

        Returns:
        y: array [Ly x Lx x 3]  y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability
        style: array [64]1D array summarizing the style of the image,if tiled it is averaged over tiles

        """
        if imgs.ndim==4:
            # make image Lz x nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (0,3,1,2))
            detranspose = (0,2,3,1)
        else:
            # make image nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (2,0,1))
            detranspose = (1,2,0)

        # pad image for net so Ly and Lx are divisible by 4
        imgs, ysub, xsub = transforms.pad_image_ND(imgs)
        # slices from padding
        slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
        slc[-2] = slice(ysub[0], ysub[-1]+1)
        slc[-1] = slice(xsub[0], xsub[-1]+1)
        slc = tuple(slc)

        # run network
        if tile or augment or imgs.ndim==4:
            y,style = self._run_tiled(imgs, augment=augment, bsize=bsize, tile_overlap=tile_overlap)
        else:
            imgs = np.expand_dims(imgs, axis=0)
            y,style = self.network(imgs)
            y, style = y[0], style[0]
        style /= (style**2).sum()**0.5

        # slice out padding
        y = y[slc]

        # transpose so channels axis is last again
        y = np.transpose(y, detranspose)

        return y, style

    def _run_tiled(self, imgi, augment=False, bsize=224, tile_overlap=0.1):
        """ run network in tiles of size [bsize x bsize]convert_images.First image is split into overlapping tiles of size [bsize x bsize].
        If augment, tiles have 50% overlap and are flipped at overlaps.
        The average of the network output over tiles is returned.
        Args:
        imgi(array): array [nchan x Ly x Lx]
        augment(bool): default False.tiles image with overlapping tiles and flips overlapped regions to augment
        bsize(int):  default 224.size of tiles to use in pixels [bsize x bsize]
        tile_overlap(float): default 0.1.fraction of overlap of tiles when computing flows

        Returns:
        yf(array): array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx].yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability
        styles(array): 1D array summarizing the style of the image, averaged over tiles

        """

        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize,
                                                            augment=augment, tile_overlap=tile_overlap)
        ny, nx, nchan, ly, lx = IMG.shape
        IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
        batch_size = self.batch_size
        niter = int(np.ceil(IMG.shape[0] / batch_size))
        y = np.zeros((IMG.shape[0], self.nclasses, ly, lx))
        for k in range(niter):
            irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
            y0, style = self.network(IMG[irange])
            y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])
            if k==0:
                styles = style[0]
            styles += style.sum(axis=0)
        styles /= IMG.shape[0]
        if augment:
            y = np.reshape(y, (ny, nx, self.nclasses, bsize, bsize))
            y = transforms.unaugment_tiles(y, self.unet)
            y = np.reshape(y, (-1, self.nclasses, bsize, bsize))

        yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
        yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
        styles /= (styles**2).sum()**0.5
        return yf, styles


