import argparse
import logging
import os
import random
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import zarr
from bfio import BioReader

import models
import utils

urls = [
    'https://www.cellpose.org/models/cytotorch_0',
    'https://www.cellpose.org/models/cytotorch_1',
    'https://www.cellpose.org/models/cytotorch_2',
    'https://www.cellpose.org/models/cytotorch_3',
    'https://www.cellpose.org/models/nucleitorch_0',
    'https://www.cellpose.org/models/nucleitorch_1',
    'https://www.cellpose.org/models/nucleitorch_2',
    'https://www.cellpose.org/models/nucleitorch_3', ]


def download_model_weights(pretrained_model, urls=urls):
    """ Downloading model weights  based on segmentation
    Args:
        pretrained_model(str): Cyto/nuclei Segmentation
        urls(list): List of urls for model weights
    """
    # cellpose directory
    start = 0
    end = len(urls)
    if pretrained_model == 'cyto':
        end += 3
    else:
        start += 4
    urls = urls[start:end]
    cp_dir = Path.home().joinpath('.cellpose')
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


def read(inpDir, flow_path, image_names):
    """ Reads vector field and unlabelled images
    Args:
        inpDir(string): Path to unlabeled images
        flow_path(array[float32]): Path to zarr file containing vector field of labeled images
        image_names(list): List of image names
    Returns:
        image_all(list): List of unlabelled image arrays
        flow_list(list): List of vector filed for labeled images

    """

    flow_list = []
    image_all = []
    root = zarr.open(str(Path(flow_path).joinpath('flow.zarr')), mode='r')

    for f in image_names:
        br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
        mask_name = str(str(f).split('.', 1)[0] + '.' + str(f).split('.', 1)[1])

        if mask_name not in root.keys():
            logger.info('%s not present in zarr file' % mask_name)
            sys.exit()
        image_all.append(np.squeeze(br.read().astype(np.float32)))

        lbl = np.squeeze(root[mask_name]['lbl'])
        vec = np.squeeze(root[mask_name]['vector'])

        vec_final = np.concatenate((vec[:, :, 2:3], vec[:, :, 0:2]), axis=2).astype(np.float32)
        cont = np.concatenate((lbl[:, :, np.newaxis], vec_final), axis=2).astype(np.float32)
        cont = cont.transpose((2, 0, 1))

        flow_list.append(cont)

    return image_all, flow_list


if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')

    # Input arguments

    parser.add_argument('--diameter', dest='diameter', type=float, default=30., help='Diameter',
                        required=False)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Unlabelled image collection to be processed by this plugin',
                        required=True)
    parser.add_argument('--pretrainedModel', dest='pretrainedModel', type=str,
                        help='Pretrained model(cyto/nuclei) or custom model path', required=False)
    parser.add_argument('--flowPath', help='Flow path should be a zarr file', type=str,
                        required=True)
    parser.add_argument('--learningRate', required=False,
                        default=0.2, type=float, help='Learning rate')
    parser.add_argument('--nEpochs', required=False,
                        default=500, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', required=False,
                        default=8, type=int, help='Batch size')
    parser.add_argument('--residualOn', required=False,
                        default=True, type=bool, help='Use residual connections')
    parser.add_argument('--styleOn', required=False,
                        default=True, type=bool, help='Use style vector')
    parser.add_argument('--concatenation', required=False,
                        default=False, type=bool,
                        help='Concatenate downsampled layers with upsampled layers (off by default which means they are added)')
    parser.add_argument('--trainFraction', required=False,
                        default=0.8, type=float, help='Test train split')

    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    # Parse the arguments
    args = parser.parse_args()  #
    diameter = args.diameter
    logger.info('diameter = {}'.format(diameter))
    inpDir = args.inpDir
    if Path.is_dir(Path(args.inpDir).joinpath('images')):
        # switch to images folder if present
        inpDir = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    pretrained_model = args.pretrainedModel
    logger.info('pretrained_model = {}'.format(pretrained_model))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    flow_path = args.flowPath

    train_fraction = args.trainFraction

    if pretrained_model == 'cyto' or pretrained_model == 'nuclei':
        torch_str = 'torch'
        download_model_weights(pretrained_model)
        model_dir = Path.home().joinpath('.cellpose', 'models')
        cpmodel_path = os.fspath(model_dir.joinpath('%s%s_0' % (pretrained_model, torch_str)))
        if pretrained_model == 'cyto':
            szmean = 30.
        else:
            szmean = 17.
    elif pretrained_model is not None:
        cpmodel_path = os.fspath(pretrained_model)
        szmean = 30

    if pretrained_model is None or not Path(cpmodel_path).exists():
        cpmodel_path = False
        logger.info('Training from scratch')
        if diameter == 0:
            rescale = False
            logger.info('Median diameter set to 0. No rescaling during training')
        else:
            rescale = True
            szmean = diameter
    else:
        rescale = True
        diameter = szmean
        logger.info('Pretrained model is being used')
        args.residualOn = 1
        args.styleOn = 1
        args.concatenation = 0

    model = models.CellposeModel(pretrained_model=cpmodel_path, model_type=pretrained_model,
                                 diam_mean=szmean, residual_on=args.residualOn,
                                 style_on=args.styleOn, concatenation=args.concatenation)
    # Surround with try/finally for proper error catching
    try:

        logger.info('Initializing ...')

        channels = [0, 0]

        image_names = [f.name for f in Path(inpDir).iterdir() if
                       f.is_file() and "".join(f.suffixes) == '.ome.tif']

        # Shuffle of images for test train split
        random.shuffle(image_names)
        idx = int(train_fraction * len(image_names))
        train_img_names = image_names[0:idx]
        test_img_names = image_names[idx:]
        logger.info('Running cellpose on %d train images  %d test images' % (
            len(train_img_names), len(test_img_names)))
        diameter = args.diameter
        logger.info('Using diameter %0.2f for all images' % diameter)

        # Read train data
        train_images, train_labels = read(inpDir, flow_path, train_img_names)

        # Read test data
        test_images, test_labels = read(inpDir, flow_path, test_img_names)

        cpmodel_path = model.train(train_images, train_labels, train_files=train_img_names,
                                   test_data=test_images, test_labels=test_labels,
                                   test_files=test_img_names,
                                   learning_rate=args.learningRate, channels=channels,
                                   save_path=outDir, rescale=rescale,
                                   n_epochs=args.nEpochs,
                                   batch_size=args.batchSize)
        model.pretrained_model = cpmodel_path
        logger.info('Model trained and saved to %s' % cpmodel_path)

    finally:
        logger.info('Closing the plugin')
        # Exit the program
        sys.exit()
