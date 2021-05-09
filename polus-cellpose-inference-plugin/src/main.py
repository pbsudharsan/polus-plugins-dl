import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import zarr
from bfio import BioReader

import models
from utils import download_url_to_file

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


def download_model_weights(pretrained_model, urls=urls):
    """ Downloading model weights  baimreadsed on segmentation
    Args:
        pretrained_model(str): Cyto/nuclei Segementation
        urls(list): list of urls for model weights

    """
    # cellpose directory
    start = 0
    end = len(urls)
    if pretrained_model == 'cyto':
        end += 4
    else:
        start += 5
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
            download_url_to_file(url, cached_file, progress=True)


TILE_SIZE = 1024
TILE_OVERLAP = 512


def main():
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')

    # Input arguments
    parser.add_argument('--diameter', dest='diameter', type=float, default=30.,
                        help='cell diameter, if 0 cellpose will estimate for each image',
                        required=False)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--pretrainedModel', dest='pretrainedModel', type=str,
                        help='model to use', required=False)

    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    # Parse the arguments
    args = parser.parse_args()
    logger.info('diameter = {}'.format(args.diameter))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        inpDir = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    pretrained_model = args.pretrainedModel
    logger.info('pretrained model = {}'.format(pretrained_model))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # Surround with try/finally for proper error catching
    try:
        logger.info('Initializing ...')
        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if
                        f.is_file() and "".join(f.suffixes) == '.ome.tif']
        rescale = None

        if pretrained_model in ['cyto', 'nuclei']:
            logger.info('Running the images on %s model' % str(pretrained_model))
            download_model_weights(pretrained_model)
            model = models.Cellpose(model_type=pretrained_model)
        elif Path(pretrained_model).exists():
            model = models.CellposeModel(pretrained_model=pretrained_model)

        else:
            raise FileNotFoundError()

        if args.diameter == 0:
            if pretrained_model in ['cyto', 'nuclei']:
                diameter = None
                logger.info('Estimating diameter for each image')
            else:
                logger.info('Using user-specified model, no auto-diameter estimation available')
                diameter = model.diam_mean
        else:
            diameter = args.diameter
            logger.info('Using diameter %0.2f for all images' % diameter)

        root = zarr.group(store=str(Path(outDir).joinpath('flow.zarr')))
        for f in inpDir_files:
            # Loop through files in inpDir image collection and process
            br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
            #  tile_size = min(TILE_SIZE,br.X)
            logger.info('Processing image %s ', f)

            # Saving pixel locations and probablity  as zarr datasets and metadata as string
            cluster = root.create_group(f)
            init_cluster_1 = cluster.create_dataset('vector', shape=(br.Y, br.X, br.Z, 3, 1),
                                                    chunks=(TILE_SIZE, TILE_SIZE, 1, 3, 1),
                                                    dtype=np.float32)
            cluster.attrs['metadata'] = str(br.metadata)
            # Iterating through z slices
            for z in range(br.Z):
                # Iterating based on tile size
                for x in range(0, br.X, TILE_SIZE):

                    for y in range(0, br.Y, TILE_SIZE):
                        x_min = max([0, x - TILE_OVERLAP])
                        x_max = min([br.X, x + TILE_SIZE + TILE_OVERLAP])

                        y_min = max([0, y - TILE_OVERLAP])
                        y_max = min([br.Y, y + TILE_SIZE + TILE_OVERLAP])

                        tile_img = br[y_min:y_max, x_min:x_max, z:z + 1, 0, 0].squeeze()
                        logger.info('Calculating flows on slice %d tile(y,x) %d :%d %d:%d ', z, y,
                                    y_max, x, x_max)
                        prob = model.eval(tile_img, diameter=diameter, rescale=rescale)

                        x_overlap = x - x_min
                        x_min = x
                        x_max = min([br.X, x + TILE_SIZE])

                        y_overlap = y - y_min
                        y_min = y
                        y_max = min([br.Y, y + TILE_SIZE])

                        prob = prob[np.newaxis,]
                        logger.info('Writing the vector field of  slice %d tile(y,x) %d :%d %d:%d ',
                                    z, y, y_max, x, x_max)
                        prob = prob[..., np.newaxis]

                        prob = prob.transpose((1, 2, 0, 3, 4))
                        root[f]['vector'][y_min:y_max, x_min:x_max, z:z + 1, 0:3, 0:1] = prob[
                                                                                         y_overlap:y_max - y_min + y_overlap,
                                                                                         x_overlap:x_max - x_min + x_overlap,
                                                                                         ...]
                        del prob

    except FileNotFoundError:
        logger.info('ERROR: model path missing or incorrect %s' % str(pretrained_model))
    finally:
        # Close the plugin regardless of successful completion
        logger.info('Closing the plugin')
        # Exit the program
        sys.exit()


if __name__ == '__main__':
    main()
