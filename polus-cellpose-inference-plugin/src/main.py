from bfio import BioReader
import argparse, logging, sys
import numpy as np
from pathlib import Path
import zarr
import models

if __name__=="__main__":
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
    parser.add_argument('--pretrained_model', dest='pretrained_model', type=str,
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
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    pretrained_model = args.pretrained_model
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

        if not pretrained_model:
            logger.info('Running the images on Cyto model')
            pretrained_model = 'cyto'
        if pretrained_model is 'cyto' or 'nuclei':
           # model = models.Cellpose(device=device,gpu=use_gpu, model_type=pretrained_model)
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
                image = np.squeeze(br.read())
                logger.info('Processing image %s ',f)
                # Serially iterating   z stack images
                if len(image.shape) >= 3:
                    if len(image.shape) == 4:
                        np.moveaxis(image, 2, 3)
                    prob_final = []
                    location_final = []
                    for i in range(image.shape[-1]):
                         prob = model.eval(image[:, :, i], diameter=diameter,rescale=rescale)
                         prob_final.append(prob.tolist())
                #        location_final.append(location.tolist())
                    prob = np.asarray(prob_final)
         #           location = np.asarray(location_final)

               # Segmenting  Greyscale images
                elif len(image.shape) == 2:
                     prob = model.eval(image, diameter=diameter,rescale=rescale)

              # Saving pixel locations and probablity in a zarr file
                cluster = root.create_group(f)
                init_cluster_1 = cluster.create_dataset('vector', shape=prob.shape, data=prob)
                cluster.attrs['metadata'] = str(br.metadata)
                del  prob

        except FileExistsError:
            logger.info('Zarr file exists. Delete the existing file %r' % str((Path(outDir).joinpath('location.zarr'))))
    except FileNotFoundError:
        logger.info('ERROR: model path missing or incorrect %s' % str(pretrained_model))
    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing the plugin')
        # Exit the program
        sys.exit()