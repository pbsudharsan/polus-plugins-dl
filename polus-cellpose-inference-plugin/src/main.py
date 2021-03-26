from bfio import BioReader
import argparse, logging, sys ,os
from urllib.parse import urlparse
from utils import download_url_to_file
import numpy as np
from pathlib import Path
import zarr
import models

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


def download_model_weights(pretrained_model,urls=urls):
    """ Downloading model weights  based on segmentation
    Args:
        pretrained_model(str): Cyto/nuclei Segementation
        urls(list): list of urls for model weights

    """
    # cellpose directory
    start= 0
    end=len(urls)
    if pretrained_model=='cyto':
        end+=4
    else:
        start+=5
    urls=urls[start:end]
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
    parser.add_argument('--diameter', dest='diameter', type=float,default=30.,help='Diameter', required=False)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--pretrainedModel', dest='pretrainedModel', type=str,default='cyto',
                        help='Filename pattern used to separate data', required=False)

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
    logger.info('pretrained_model = {}'.format(pretrained_model))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # Surround with try/finally for proper error catching
    try:
        logger.info('Initializing ...')
        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes) == '.ome.tif']
        rescale = None

        if args.diameter == 0:
            diameter = None
            logger.info('Estimating diameter for each image')
        else:
            diameter = args.diameter
            logger.info(' Using diameter %0.2f for all images' % diameter)

        if pretrained_model is 'cyto' or 'nuclei':
             logger.info('Running the images on %s model'% str(pretrained_model))
             download_model_weights(pretrained_model)
             model = models.Cellpose( model_type=pretrained_model)
        elif Path(pretrained_model).exists():
            model = models.CellposeModel( pretrained_model=pretrained_model)
            rescale = model.diam_mean / diameter
        else:
            raise FileNotFoundError()

        try:
            if Path(outDir).joinpath('flow.zarr').exists():
                raise FileExistsError()

            root = zarr.group(store=str(Path(outDir).joinpath('flow.zarr')))
            for f in inpDir_files:
                # Loop through files in inpDir image collection and process
                br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
                tile_size = min(1024,br.X)
                logger.info('Processing image %s ',f)
           #     out_image=np.zeros((br.Z,br.X,br.Y,3)).astype(np.float32)
                out_image = np.zeros((1, tile_size, tile_size, 3)).astype(np.float32)
                # Saving pixel locations and probablity in a zarr file
                cluster = root.create_group(f)
                init_cluster_1 = cluster.create_dataset('vector', shape=(br.Y,br.X,br.Z,3,1),
                                                        chunks=(tile_size, tile_size, 1, 1, 1), dtype=out_image.dtype)
                cluster.attrs['metadata'] = str(br.metadata)
                # Iterating through z slices
                for z in range(br.Z):
                    # Iterating based on tile size
                    for x in range(0, br.X, tile_size):
                        x_max = min([br.X, x + tile_size])
                        for y in range(0, br.Y, tile_size):
                            y_max = min([br.Y, y + tile_size])
                            tile_img = (br[y:y_max, x:x_max,z:z+1, 0,0]).squeeze()
                            logger.info('Calculating flows on slice %d tile(y,x) %d :%d %d:%d ',z,y,y_max,x,x_max)
                            prob = model.eval(tile_img, diameter=diameter,rescale=rescale)
                            prob=prob[np.newaxis,]
     #                       out_image[z:z+1,y:y_max, x:x_max,] = prob[np.newaxis,]
                            logger.info('Shaping array as per ome format')
                            out_image= out_image[...,np.newaxis]
                            out_image=out_image.transpose((1,2,0,3,4)).astype(np.float32)
                            print(init_cluster_1.shape,out_image.shape)
                            name=str(f)+'vector'
                            init_cluster_1[name]=out_image


                            del prob, out_image

        except FileExistsError:
            logger.info('Zarr file exists. Delete the existing file %r' % str((Path(outDir).joinpath('location.zarr'))))
    except FileNotFoundError:
        logger.info('ERROR: model path missing or incorrect %s' % str(pretrained_model))
    finally:
        # Close the plugin regardless of successful completion
        logger.info('Closing the plugin')
        # Exit the program
        sys.exit()

if __name__ == '__main__':
    main()