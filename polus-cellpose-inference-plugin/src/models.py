# Code sourced code from Cellpose repo https://github.com/MouseLand/cellpose/tree/master/cellpose

import logging
import os
import pathlib
import time
import cv2
import numpy as np
from tqdm import trange
import dynamics
import transforms
import utils
from core import UnetModel, assign_device, convert_images, parse_model_string

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("model")
logger.setLevel(logging.INFO)

model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')


def dx_to_circ(dP):
    """ Converts flows to vector field

    This function converts flow representation to vector field.

    Args:
        dP (array) : Flow representation.
    Returns:
        flow(array): Vector field.

    """

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


class Cellpose():
    """ Main class for Cellpose model segmentation

    This is the main class which handles Cellpose segmentation model.

    Args:
        model_type(str): Type of model.
        net_avg(bool): Default True. Loads the 4 built-in networks and averages them if True, loads one network if False.
        torch(bool): Default False. Run model using torch if available.

    """

    def __init__(self, model_type='cyto', net_avg=True, torch=True):

        super(Cellpose, self).__init__()
        self.torch = torch
        torch_str = ['', 'torch'][self.torch]
        # assign device (GPU or CPU)
        device, gpu = assign_device(self.torch)
        self.device = device
        self.gpu = gpu
        model_type = 'cyto' if model_type is None else model_type
        self.pretrained_model = [
            os.fspath(model_dir.joinpath('%s%s_%d' % (model_type, torch_str, j))) for j in
            range(4)]
        self.pretrained_size = os.fspath(
            model_dir.joinpath('size_%s%s_0.npy' % (model_type, torch_str)))
        self.diam_mean = 30. if model_type == 'cyto' else 17.
        if not net_avg:
            self.pretrained_model = self.pretrained_model[0]
        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean, torch=self.torch)
        self.cp.model_type = model_type
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

    def eval(self, x, batch_size=8, channels=None, invert=False, normalize=True, diameter=30.,
             anisotropy=None, tile_size=224,
             net_avg=True, augment=False, tile=True, tile_overlap=0.1, resample=False, interp=True,
             flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
             stitch_threshold=0.0, rescale=None):
        """ Runs cellpose and get masks

        This function handles function call to network to compute masks.

        Args:
            x(array): 2d Image
            batch_size(int): Default 8. Number of 224x224 patches to run simultaneously on the GPU
            channels(list): List of channels, either of length 2 or of length number of images by 2
            invert(bool): Invert image pixel intensity before running network (if True, image is also normalized)
            normalize(bool): Normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
            diameter(float): Default 30. If set to None, then diameter is automatically estimated if size model is loaded
            anisotropy(float): Rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)
            net_avg(bool): Default True. Runs the 4 built-in networks and averages them if True, runs one network if False
            augment(bool): Default False. Tiles image with overlapping tiles and flips overlapped regions to augment
            tile(bool): Default True. Tiles image to ensure GPU/CPU memory usage limited
            tile_overlap(float): Default 0.1. Fraction of overlap of tiles when computing flows
            resample(bool): Default False. Run dynamics at original image size (will be slower but create more accurate boundaries)
            interp(bool): Default True. Interpolate during 2D dynamics
            flow_threshold(float): Default 0.4. Flow error threshold (all cells with errors below threshold are kept)
            cellprob_threshold(float): Default 0.0. Cell probability threshold (all pixels with prob above threshold kept for masks)
            min_size(int): Default 15. Minimum number of pixels per mask
            stitch_threshold(float): Default 0.0. If stitch_threshold>0.0  and equal image sizes
            rescale(float): Default None if diameter is set to None, and rescale is not None, then rescale is used instead of diameter for resizing image
        Returns:
            masks(arrays): list of 2D Labelled image where 0=no masks; 1,2,...=mask labels
            flows(array): list of lists 2D arrays, flows[k][0] = XY flow in HSV 0-255, flows[k][1] = Flows at each pixel and flows[k][2] = Cell probability centered at 0.0
            styles(list): Style vector summarizing each image, also used to estimate size of objects in image
            diams[list]: List of diameters

        """

        if not isinstance(x, list):
            if x.ndim < 2 or x.ndim > 5:
                raise ValueError('%dD images not supported' % x.ndim)
            else:
                x = [x]
        else:
            for xi in x:
                if xi.ndim < 2 or xi.ndim > 5:
                    raise ValueError('%dD images not supported' % xi.ndim)

        nimg = len(x)
        # make rescale into length of x
        if diameter is not None and not (not isinstance(diameter, (list, np.ndarray)) and
                                         (diameter == 0 or (
                                                 diameter == 30. and rescale is not None))):
            if not isinstance(diameter, (list, np.ndarray)) or len(diameter) == 1 or len(
                    diameter) < nimg:
                diams = diameter * np.ones(nimg, np.float32)
            else:
                diams = diameter
            rescale = self.diam_mean / diams
        else:
            if rescale is not None and (
                    not isinstance(rescale, (list, np.ndarray)) or len(rescale) == 1):
                rescale = rescale * np.ones(nimg, np.float32)

            if self.pretrained_size is not None and rescale is None:

                tic = time.time()
                diams, _ = self.sz.eval(x, channels=channels, invert=invert, batch_size=batch_size,
                                        augment=augment, tile=tile)
                rescale = self.diam_mean / diams
                logger.info('Estimated cell diameters  in %0.2f sec' % (time.time() - tic))
                logger.info('Diameter(s) = {}'.format(diams))
            else:
                if rescale is None:
                    rescale = np.ones(nimg, np.float32)
                diams = self.diam_mean / rescale
        prob = self.cp.eval(x, batch_size=batch_size,
                            invert=invert,
                            rescale=rescale,
                            anisotropy=anisotropy,
                            channels=channels,
                            augment=augment,
                            tile=tile,
                            net_avg=net_avg,
                            tile_overlap=tile_overlap,
                            resample=resample,
                            interp=interp,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            min_size=min_size,
                            stitch_threshold=stitch_threshold, tile_size=tile_size)
        return prob


class CellposeModel(UnetModel):
    """ Class for the segmenting masks

    Class for cellpose model.

    Args:
        pretrained_model(str/list): Path to pretrained cellpose model(s), if False, no model loaded;if None, built-in 'cyto' model loaded
        net_avg(bool): Default True. Loads the 4 built-in networks and averages them if True, loads one network if False
        diam_mean(float): Default 30. If set to None, then diameter is automatically estimated if size model is loaded
        device(torch): Set GPU/CPU
        residual_on(bool): Network parameter
        style_on(bool): Network parameter
        concatenation(bool): Network parameter

    """

    def __init__(self, pretrained_model=False, torch=True, gpu=None,
                 diam_mean=30., net_avg=True, device=None,
                 residual_on=True, style_on=True, concatenation=False):

        torch = True
        self.torch = torch
        if not device:
            device, gpu = assign_device(self.torch)
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
        super().__init__(pretrained_model=False,
                         diam_mean=diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                         gpu=gpu,
                         nclasses=nclasses, torch=torch)
        self.unet = False
        self.pretrained_model = pretrained_model
        if self.pretrained_model is not None and isinstance(self.pretrained_model, str):
            self.net.load_model(self.pretrained_model, cpu=(not self.gpu))
        ostr = ['off', 'on']
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])


    def eval(self, imgs, batch_size=8, channels=None, normalize=True, invert=False,
             rescale=None, diameter=None, anisotropy=None, net_avg=True,
             augment=False, tile=True, tile_overlap=0.1, tile_size=224,
             resample=False, interp=True, flow_threshold=0.4, cellprob_threshold=0.0,
             compute_masks=False,
             min_size=15, stitch_threshold=0.0, not_compute=True):
        """ Segment list of images

        This function handles function calls to neural network and mask computation.

        Args:
            imgs(array): Can be list of 2D/3D/4D images, or array of 2D/3D images
            batch_size(int): Default 8. Number of 224x224 patches to run simultaneously on the GPU
            channels(list): Default None. List of channels, either of length 2 or of length number of images by 2
            normalize(bool): Default True. Normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
            invert(bool):  Default False. Invert image pixel intensity before running network
            rescale(float): Resize factor for each image, if None, set to 1.0
            diameter(float): Diameter for each image (only used if rescale is None), if diameter is None, set to diam_mean
            net_avg(bool): Default True. Runs the 4 built-in networks and averages them if True, runs one network if False
            augment(bool): Default False. Tiles image with overlapping tiles and flips overlapped regions to augment
            tile(bool): Default True. Tiles image to ensure GPU/CPU memory usage limited (recommended)
            tile_overlap(float): Default 0.1. Fraction of overlap of tiles when computing flows
            resample(bool): Default False. Run dynamics at original image size (will be slower but create more accurate boundaries)
            interp(bool): Default True. Interpolate during 2D dynamics
            flow_threshold(float): Default=.4. Flow error threshold (all cells with errors below threshold are kept)
            cellprob_threshold(float): Cell probability threshold (all pixels with prob above threshold kept for masks)
            compute_masks(bool): Default True. Whether or not to compute dynamics and return masks
            min_size(int): Default 15. Minimum number of pixels per mask, can turn off with -1
        Returns:
            masks(list): List of 2D arrays.labelled image, where 0=no masks; 1,2,...=mask labels.
            flows(list): List of lists 2D arrays. flows[k][0] = XY flow in HSV 0-255, flows[k][1] = flows at each pixel, flows[k][2] = the cell probability centered at 0.0
            styles(list): Style vector summarizing each image, also used to estimate size of objects in image

        """

        x, nolist = convert_images(imgs.copy(), channels, normalize, invert)
        nimg = len(x)
        self.batch_size = batch_size
        styles = []
        flows = []
        masks = []
        if rescale is None:
            if diameter is not None:
                if not isinstance(diameter, (list, np.ndarray)):
                    diameter = diameter * np.ones(nimg)
                rescale = self.diam_mean / diameter
            else:
                rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)
        iterator = trange(nimg) if nimg > 1 else range(nimg)
        if isinstance(self.pretrained_model, list) and not net_avg:
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
        flow_time = 0
        net_time = 0
        for i in iterator:
            img = x[i].copy()
            Ly, Lx = img.shape[:2]
            tic = time.time()
            shape = img.shape
            # rescale image for flow computation
            img = transforms.resize_image(img, rsz=rescale[i])
            y, style = self._run_nets(img, net_avg=net_avg,
                                      augment=augment, tile=tile,
                                      tile_overlap=tile_overlap, bsize=tile_size)
            net_time += time.time() - tic
            styles.append(style)
            if compute_masks:
                tic = time.time()
                if resample:
                    y = transforms.resize_image(y, shape[-3], shape[-2])
                cellprob = y[:, :, -1]
                dP = y[:, :, :2].transpose((2, 0, 1))
                niter = 1 / rescale[i] * 200
                p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                                          niter=niter, interp=interp, use_gpu=self.gpu)
                maski = dynamics.get_masks(p, iscell=(cellprob > cellprob_threshold),
                                           flows=dP, threshold=flow_threshold)
                maski = utils.fill_holes_and_remove_small_masks(maski)
                maski = transforms.resize_image(maski, shape[-3], shape[-2],
                                                interpolation=cv2.INTER_NEAREST)
                flows.append([dx_to_circ(dP), dP, cellprob, p])
                masks.append(maski)
                flow_time += time.time() - tic
                return y, masks, styles
            else:
                if not_compute:
                    y = transforms.resize_image(y, shape[-3], shape[-2])
                else:
                    flows.append([None] * 3)
                    masks.append([])
                return y


class SizeModel():
    """ Class for regression model

    This class handles linear regression model which is used for determining the size of objects in image
    used to rescale before input to cp_model. Styles from cp_model are used.

    Args:
        cp_model: Model from which to get styles
        device(Torch device):  Set cpu/gpu usage
        pretrained_size(str): Path to pretrained size model

    """

    def __init__(self, cp_model, device=None, pretrained_size=None, **kwargs):

        super(SizeModel, self).__init__(**kwargs)
        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.device = self.cp.device
        self.diam_mean = self.cp.diam_mean
        self.torch = self.cp.torch
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params['diam_mean']
        if not hasattr(self.cp, 'pretrained_model'):
            raise ValueError('provided model does not have a pretrained_model')

    def eval(self, imgs=None, styles=None, channels=None, normalize=True, invert=False,
             augment=False, tile=True,
             batch_size=8):
        """ Produce style or use style input to predict size of objects in image

        This function does object size estimation is done in two steps:
        1. Use a linear regression model to predict size from style in image.
        2. Resize image to predicted size and run CellposeModel to get output masks.
        Take the median object size of the predicted masks as the final predicted size.

        Args:
            imgs(array): Array of 2D images.
            styles(array/list): Styles for images x - if x is None then styles must not be None
            channels(list): List of channels, either of length 2 or of length number of images by 2
            normalize(bool): Default True. Normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
            invert(bool): Default False. Invert image pixel intensity before running network
            augment(bool): Default False. Tiles image with overlapping tiles and flips overlapped regions to augment
            tile(bool): Default True. Tiles image to ensure GPU/CPU memory usage limited (recommended)
        Returns:
            diam(array[float]): Final estimated diameters from images x or styles style after running both steps
            diam_style(array[float]): Estimated diameters from style alone

        """

        if imgs is not None:
            x, nolist = convert_images(imgs.copy(), channels, normalize, invert)
            nimg = len(x)
        if styles is None:
            logger.info(f'computing styles from images')
            styles = self.cp.eval(x, net_avg=False, augment=augment, tile=tile, compute_masks=True,
                                  not_compute=False)[
                -1]
            diam_style = self._size_estimation(np.array(styles))
        else:
            styles = np.array(styles) if isinstance(styles, list) else styles
            diam_style = self._size_estimation(styles)
        diam_style[np.isnan(diam_style)] = self.diam_mean
        if imgs is not None:
            masks = self.cp.eval(x, rescale=self.diam_mean / diam_style, net_avg=False,
                                 augment=augment, tile=tile, interp=False, compute_masks=True)[-2]
            diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
            diam[diam == 0] = self.diam_mean
            diam[np.isnan(diam)] = self.diam_meanl
        else:
            diam = diam_style
            logger.info('No images provided, using diameters estimated from styles alone')
        if nolist:
            return diam[0], diam_style[0]
        else:
            return diam, diam_style

    def _size_estimation(self, style):
        """ Linear regression from style to size

        This function computes sizes using "diameters" from square estimates not circles. Therefore a conversion factor is included.

        Args:
            style(array) : Style of Unet model
        Returns:
            szest(float): Diameter estimateof cells/nuclei

        """

        szest = np.exp(self.params['A'] @ (style - self.params['smean']).T +
                       np.log(self.diam_mean) + self.params['ymean'])
        szest = np.maximum(5., szest)
        return szest
