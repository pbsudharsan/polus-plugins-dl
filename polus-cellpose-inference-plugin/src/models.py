'''

Most of the sourced  code is so from Cellpose repo  https://github.com/MouseLand/cellpose/tree/master/cellpose

'''
import os, sys, time,  pathlib
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse


from mxnet import gluon, nd
import mxnet as mx
import logging
import transforms, dynamics, utils, resnet_style

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("models")
logger.setLevel(logging.INFO)

class Cellpose():
    """ main model which combines SizeModel and CellposeModel """


    def __init__(self, gpu=False, model_type='cyto', net_avg=True, batch_size=8, device=None):
        """
        Args:
        gpu  (optional[bool]): Default False. Whether or not to save model to GPU, will check if GPU available
        model_type(optional[str]) :  Default 'cyto' . 'cyto'=cytoplasm model; 'nuclei'=nucleus model
        net_avg (optional[ bool],):  Default True. Loads the 4 built-in networks and averages them if True, loads one network if False
        batch_size(optional[int]) : Default 8 .Number of 224x224 patches to run simultaneously on the GPU(can make smaller or bigger depending on GPU memory usage)
        device (optional[mxnet device]) : Where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,recommended if you want to use a specific GPU (e.g. mx.gpu(4))

        """
        super(Cellpose, self).__init__()
        # assign device (GPU or CPU)
        if device is not None:
            self.device = device
        elif gpu and utils.use_gpu():
            self.device = mx.gpu()
            logger.info(' using GPU')
        else:
            self.device = mx.cpu()
            logger.info(' using CPU')

        self.batch_size=batch_size
        model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')
        if model_type is None:
            model_type = 'cyto'

        self.pretrained_model = [os.fspath(model_dir.joinpath('%s_%d'%(model_type,j))) for j in range(4)]
        self.pretrained_size = os.fspath(model_dir.joinpath('size_%s_0.npy'%(model_type)))
        if model_type=='cyto':
            self.diam_mean = 30.
        else:
            self.diam_mean = 17.
        if not os.path.isfile(self.pretrained_model[0]):
            download_model_weights()
        if not net_avg:
            self.pretrained_model = self.pretrained_model[0]

        self.cp = CellposeModel(device=self.device,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean, 
                                batch_size=self.batch_size)

        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)

    def eval(self, x,image_name, diameter=30., invert=False,  anisotropy=None,
             net_avg=True, augment=True, tile=True,compute_masks=True,rescale=None):
        """ run cellpose and get masks

        Args :
        x ( array) :  Array of 2D image.
        diameter (optional[float]):  Default 30. If set to None, then diameter is automatically estimated if size model is loaded.
        invert (optional[bool]) :  default False. Invert image pixel intensity before running network.
        anisotropy (optional[float]): Default None. For 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y).
        net_avg (optional[bool]): Default True.Runs the 4 built-in networks and averages them if True, runs one network if False.
        augment(optional[bool] ) :Tiles image with overlapping tiles and flips overlapped regions to augment.
        tile (optional[bool]): Default True. Default True tiles image to ensure GPU/CPU memory usage limited (recommended).
        flow_threshold (optional[float ]) : Default 0.4. Flow error threshold (all cells with errors below threshold are kept) (not used for 3D).
        cellprob_threshold(optional[float]): Default 0.0. Cell probability threshold (all pixels with prob above threshold kept for masks).
        rescale(optional[float] ): Default None. If diameter is set to None, and rescale is not None, then rescale is used instead of diameter for resizing image.


        Returns :
        loc  (float32, 3D array ) :  Final locations of each pixel after dynamics
        prob (list of arrays) :
            flows[k][0] = XY flow in HSV 0-255
            flows[k][1] = flows at each pixel
            flows[k][2] = the cell probability centered at 0.0


        """

        if not isinstance(x,list):
            nolist = True
            x = [x]
        else:
            nolist = False

        tic0 = time.time()
        nimg = len(x)
        logger.info('processing %s image(s)'%image_name)
        # make rescale into length of x
        if diameter is not None and diameter!=0:
            if not isinstance(diameter, list) or len(diameter)==1 or len(diameter)<nimg:
                diams = diameter * np.ones(nimg, np.float32)
            else:
                diams = diameter
            rescale = self.diam_mean / diams.copy() 
        else:
            if rescale is not None and (not isinstance(rescale, list) or len(rescale)==1):
                rescale = rescale * np.ones(nimg, np.float32)
            if self.pretrained_size is not None and rescale is None :
                tic = time.time()
                diams, _ = self.sz.eval(x, invert=invert, batch_size=self.batch_size, augment=augment, tile=tile)
                rescale = self.diam_mean / diams.copy()
                logger.info('estimated cell diameter for %s image in %0.2f sec'%(image_name, time.time()-tic))
            else:
                if rescale is None:

                    rescale = np.ones(nimg, np.float32)
                diams = self.diam_mean / rescale.copy() 

        tic = time.time()
        loc,prob= self.cp.eval(x, invert=invert, rescale=rescale, anisotropy=anisotropy,augment=augment, tile=tile,net_avg=net_avg)
        logger.info('Estimated probablity of cells   for %s image in %0.2f sec'%(image_name, time.time()-tic))
        logger.info(' TOTAL TIME %0.2f sec'%(time.time()-tic0))
        return loc,prob

class CellposeModel():


    def __init__(self, gpu=False, pretrained_model=False, batch_size=8,
                    diam_mean=30., net_avg=True, device=None, unet=False):
        """

         Args :

         gpu(optional[bool]): Default False. Whether or not to save model to GPU, will check if GPU available

         pretrained_model(optional[str]) : path to pretrained cellpose model(s), if False, no model loaded;if None, built-in 'cyto' model loaded

         net_avg (optional[bool]):Default True. loads the 4 built-in networks and averages them if True, loads one network if False

         batch_size:  (optional[int]): Default 8. Number of 224x224 patches to run simultaneously on the GPU (can make smaller or bigger depending on GPU memory usage)

         diam_mean: float (optional, default 27.)
             mean 'diameter', 27. is built in value for 'cyto' model
         device (mxnet device ): Where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,recommended if you want to use a specific GPU (e.g. mx.gpu(4))

         """
        super(CellposeModel, self).__init__()
        if device is not None:
            self.device = device
        elif gpu and utils.use_gpu():
            self.device = mx.gpu()
            logger.info(' using GPU')
        else:
            self.device = mx.cpu()
            logger.info(' using CPU')

        nout = 3

        self.pretrained_model = pretrained_model
        self.batch_size=batch_size
        self.diam_mean = diam_mean

        nbase = [32,64,128,256]
        self.net = resnet_style.CPnet(nbase, nout=nout)
        self.net.hybridize(static_alloc=True, static_shape=True)
        self.net.initialize(ctx = self.device)

        model_dir = pathlib.Path.home().joinpath('.cellpose', 'models')

        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_parameters(pretrained_model)


    def eval(self, x, channels=[0,0], invert=False, rescale=None, anisotropy=None, net_avg=True, augment=True,
             tile=True,compute_masks=False,flow_threshold=0.4, cellprob_threshold=0.0):
        """
            Segment images

           Args :
            x ( array) : 2D image array
            channels (optional[list]) :
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].
            invert  (optional[bool]) : Invert image pixel intensity before running network
            rescale  (optional[float]) : Default None. Resize factor for each image, if None, set to 1.0
            net_avg(optional[bool]): Default True.Runs the 4 built-in networks and averages them if True, runs one network if False
            augment(optional[bool]): Default True. tiles image with overlapping tiles and flips overlapped regions to augment
            tile (optional[bool]: Default True.Tiles image to ensure GPU/CPU memory usage limited (recommended)
            flow_threshold(optional[float]): Default 0.4. flow error threshold (all cells with errors below threshold are kept) (not used for 3D)
            cellprob_threshold (optional[float]): default 0.0. cell probability threshold (all pixels with prob above threshold kept for masks)
            compute_masks (optional[bool]) : Default True.Whether or not to compute dynamics from numba import njit and return masks.This is set to False when retrieving the styles for the size model.


            Returns :
            masks(array) : labelled image, where 0=no masks; 1,2,...=mask labels
            flows(array):
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0
            styles(1D array): style vector summarizing each image, also used to estimate size of objects in image

        """
        nimg = len(x)
        if channels is not None:
            if len(channels)==2:

                if not isinstance(channels[0], list):
                    channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i], invert=invert) for i in range(nimg)]


        styles = []
        flows = []
        masks = []
        if rescale is None:
            rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)
        if nimg > 1:
            iterator = trange(nimg)
        else:
            iterator = range(nimg)

        if isinstance(self.pretrained_model, list) and not net_avg:
            self.net.load_parameters(self.pretrained_model[0])
            self.net.collect_params().grad_req = 'null'


        for i in iterator:

            img = x[0].copy()

            if img.shape[0]<3:
                img = np.transpose(img, (1, 2, 0))
            Ly,Lx = img.shape[:2]
            if img.shape[-1]==1:
                img = np.concatenate((img, 0.*img), axis=-1)

            if isinstance(self.pretrained_model, str) or not net_avg:
                y, style = self._run_net(img, rsz=rescale[i], augment=augment, tile=tile)
            else:
                y, style = self._run_many(img, rsz=rescale[i], augment=augment, tile=tile)

            styles.append(style)
            if not compute_masks:
                dP = np.stack((y[...,0], y[...,1]), axis=0)
                niter = 1 / rescale[i] * 200
                p = dynamics.follow_flows(-1 * dP  / 5. , niter=niter)
                return p, y
            else :
                cellprob = y[..., -1]
                dP = np.stack((y[..., 0], y[..., 1]), axis=0)
                niter = 1 / rescale[i] * 200
                p = dynamics.follow_flows(-1 * dP / 5., niter=niter)

                maski = dynamics.get_masks(p, iscell=(cellprob > cellprob_threshold),
                                           flows=dP, threshold=flow_threshold)
                maski = dynamics.fill_holes(maski)
                masks.append(maski)
                return  masks,styles


    def _run_many(self, img, rsz=1.0, augment=True, tile=True):
        """ loop over networks in pretrained_model and average results

        Args :
        img (float): [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]
        rsz(optional[ float ]): Default 1.0. resize coefficient for image
        augment (optional[bool]): default True .tiles image with overlapping tiles and flips overlapped regions to augment
        tile (optional[bool]):default True.tiles image to ensure GPU memory usage limited (recommended)


        Returns:

        yup(array):  [3 x Ly x Lx].yup is output averaged over networks;yup[0] is Y flow; yup[1] is X flow; yup[2] is cell probability
        style(array):  [64] 1D array summarizing the style of the image,if tiled it is averaged over tiles,but not averaged over networks.

        """
        for j in range(len(self.pretrained_model)):

            self.net.load_parameters(self.pretrained_model[j])
            self.net.collect_params().grad_req = 'null'
            yup0, style = self._run_net(img, rsz=rsz, augment=augment, tile=tile)

            if j==0:
                yup = yup0
            else:
                yup += yup0

        yup = yup / len(self.pretrained_model)
        return yup, style

    def  _run_net(self, imgs, rsz=1.0, augment=True, tile=True, bsize=224):
        """ run network on stack of images at the same time

        (faster if augment is False)

        Args:
        imgs(array):  [Ly x Lx x nchan]
        rsz(optional[float]): default 1.0 .resize coefficient(s) for image
        augment(optional[bool]): default True.tiles image with overlapping tiles and flips overlapped regions to augment
        tile(optional[bool]): default True .tiles image to ensure GPU/CPU memory usage limited (recommended);cannot be turned off for 3D segmentation
        bsize (optional[int]): default 224 . Size of tiles to use in pixels [bsize x bsize]

        Returns :
        y(array): [Ly x Lx x 3] y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability
        style(array):  [64] 1D array summarizing the style of the image,if tiled it is averaged over tiles

        """   
        shape = imgs.shape
        # rescale image for flow computation
        if not isinstance(rsz, list):
            rsz = [rsz, rsz]
        if (np.abs(np.array(rsz) - 1.0) < 0.03).sum() < 2:
            resize=True
            Ly = int(imgs.shape[-3] * rsz[0])
            Lx = int(imgs.shape[-2] * rsz[1])

            imgs = transforms.resize_image(imgs, Ly, Lx)
        else:
            resize=False

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
            y,style = self._run_tiled(imgs, augment=augment, bsize=bsize)
        else:
            imgs = nd.array(np.expand_dims(imgs, axis=0), ctx=self.device)
            y,style = self.net(imgs)
            y = y[0].asnumpy()
            imgs = imgs.asnumpy()
            style = style.asnumpy()[0]
        style /= (style**2).sum()**0.5

        # slice out padding
        y = y[slc]

        # transpose so channels axis is last again
        y = np.transpose(y, detranspose)
        
        # rescale back to original size
        if resize:
            y = transforms.resize_image(y, shape[-3], shape[-2])
        
        return y, style
    
    def _run_tiled(self, imgi, augment=True, bsize=224):
        """ run network in tiles of size [bsize x bsize]

        First image is split into overlapping tiles of size [bsize x bsize].
        Then 4 versions of each tile are created:
            * original
            * flipped vertically
            * flipped horizontally
            * flipped vertically and horizontally
        The average of the network output over tiles is returned.

        Args:
        imgi(array):  [nchan x Ly x Lx]
        augment(optional[bool]): default True.tiles image with overlapping tiles and flips overlapped regions to augment
        bsize (optional[int]): default 224. size of tiles to use in pixels [bsize x bsize]

        Returns
        yf(array):  [3 x Ly x Lx] or [Lz x 3 x Ly x Lx] yf is averaged over tiles. yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability
        styles(array[64]): 1D array summarizing the style of the image, averaged over tiles

        """

        if imgi.ndim==4:
            Lz, nchan = imgi.shape[:2]
            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize, augment=augment)
            ny, nx, nchan = IMG.shape[:3]
            yf = np.zeros((Lz, 3, imgi.shape[2], imgi.shape[3]), np.float32)
            styles = []
            if ny*nx > self.batch_size:
                ziterator = trange(Lz)
                for i in ziterator:
                    yfi, stylei = self._run_tiled(imgi[i], augment=augment, bsize=bsize)
                    yf[i] = yfi
                    styles.append(stylei)
            else:
                # run multiple slices at the same time
                ntiles = ny*nx
                nimgs = max(2, int(np.round(self.batch_size / ntiles)))
                niter = int(np.ceil(Lz/nimgs))
                ziterator = trange(niter)
                for k in ziterator:
                    IMGa = np.zeros((ntiles*nimgs, nchan, bsize, bsize), np.float32)
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[k*nimgs+i], bsize=bsize, augment=augment)
                        IMGa[i*ntiles:(i+1)*ntiles] = np.reshape(IMG, (ny*nx, nchan, bsize, bsize))
                    img = nd.array(IMGa, ctx=self.device)
                    y0, style = self.net(img)
                    ya = y0.asnumpy()
                    stylea = style.asnumpy()
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        y = ya[i*ntiles:(i+1)*ntiles]
                        if augment:
                            y = np.reshape(y, (ny, nx, 3, bsize, bsize))
                            y = transforms.unaugment_tiles(y)
                            y = np.reshape(y, (-1, 3, bsize, bsize))
                        yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                        yfi = yfi[:,:imgi.shape[2],:imgi.shape[3]]
                        yf[k*nimgs+i] = yfi
                        stylei = stylea[i*ntiles:(i+1)*ntiles].sum(axis=0)
                        stylei /= (stylei**2).sum()**0.5
                        styles.append(stylei)
            return yf, np.array(styles)
        else:
            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize, augment=augment)
            ny, nx, nchan = IMG.shape[:3]
            IMG = np.reshape(IMG, (ny*nx, nchan, bsize, bsize))
            nbatch = self.batch_size
            niter = int(np.ceil(IMG.shape[0]/nbatch))
            y = np.zeros((IMG.shape[0], 3, bsize, bsize))
            for k in range(niter):
                irange = np.arange(nbatch*k, min(IMG.shape[0], nbatch*k+nbatch))
                img = nd.array(IMG[irange], ctx=self.device)
                y0, style = self.net(img)
                y[irange] = y0.asnumpy()
                if k==0:
                    styles = style.asnumpy()[0]
                styles += style.asnumpy().sum(axis=0)
            
            styles /= IMG.shape[0]
            if augment:
                y = np.reshape(y, (ny, nx, 3, bsize, bsize))
                y = transforms.unaugment_tiles(y)
                y = np.reshape(y, (-1, 3, bsize, bsize))
            yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
            yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
            styles /= (styles**2).sum()**0.5
            return yf, styles


class SizeModel():
    """
    linear regression model for determining the size of objects in image used to rescale before input to Cellpose Model
        uses styles from Cellpose Model


    """
    def __init__(self, cp_model, device=mx.cpu(), pretrained_size=None, **kwargs):
        """
        Args:
        cp_model(CellposeModel): cellpose model from which to get styles
        device(optional[mxnet device]): default mx.cpu().where cellpose model is saved (mx.gpu() or mx.cpu())
        pretrained_size(str): path to pretrained size model
        """
        super(SizeModel, self).__init__(**kwargs)

        self.device = device
        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.diam_mean = self.cp.diam_mean
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params['diam_mean']

    def eval(self, x=None, style=None, channels=None, invert=False, augment=True, tile=True,
                batch_size=8):
        """ use images x to produce style or use style input to predict size of objects in image

        Object size estimation is done in two steps:
        1. use a linear regression model to predict size from style in image
        2. resize image to predicted size and run CellposeModel to get output masks.
            Take the median object size of the predicted masks as the final predicted size.

        Args:
        cp_model(CellposeModel): cellpose model from which to get styles
        device(optional[mxnet device]): default mx.cpu().Where cellpose model is saved (mx.gpu() or mx.cpu())
        pretrained_size(str): path to pretrained size model

        """
        if style is None and x is None:
            logger.info('Error: no image or features given')
            return

        nimg = len(x)
        if channels is not None:
            if len(channels)==2:
                if not isinstance(channels[0], list):
                    channels = [channels for i in range(nimg)]
            x = [transforms.reshape(x[i], channels=channels[i], invert=invert) for i in range(nimg)]
        diam_style = np.zeros(nimg, np.float32)

        
        if style is None:
            if nimg>1:
                iterator = trange(nimg)
            else:
                iterator = range(nimg)
            for i in iterator:
                img = x[i]
                style = self.cp.eval([img], net_avg=False, augment=augment, tile=tile, compute_masks=True)[-1]

                diam_style[i] = self._size_estimation(style)

        else:
            for i in range(len(style)):
                diam_style[i] = self._size_estimation(style[i])
        diam_style[diam_style==0] = self.diam_mean
        diam_style[np.isnan(diam_style)] = self.diam_mean
        masks = self.cp.eval(x, rescale=self.diam_mean/diam_style, net_avg=False, augment=augment, tile=tile, compute_masks=True)[0]
        diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
        diam[diam==0] = self.diam_mean
        diam[np.isnan(diam)] = self.diam_mean

        return diam, diam_style

    def _size_estimation(self, style):
        """ linear regression from style to size 
        
            sizes were estimated using "diameters" from square estimates not circles; 
            therefore a conversion factor is included (to be removed)
        
        """
        szest = np.exp(self.params['A'] @ (style - self.params['smean']).T +
                        np.log(self.diam_mean * (np.pi**0.5)/2) + self.params['ymean'])
        szest = np.maximum(5., szest / ((np.pi**0.5)/2))
        return szest

urls = ['http://www.cellpose.org/models/cyto_0',
        'http://www.cellpose.org/models/cyto_1',
        'http://www.cellpose.org/models/cyto_2',
        'http://www.cellpose.org/models/cyto_3',
        'http://www.cellpose.org/models/size_cyto_0.npy',
        'http://www.cellpose.org/models/nuclei_0',
        'http://www.cellpose.org/models/nuclei_1',
        'http://www.cellpose.org/models/nuclei_2',
        'http://www.cellpose.org/models/nuclei_3',
        'http://www.cellpose.org/models/size_nuclei_0.npy']


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
            utils.download_url_to_file(url, cached_file)
