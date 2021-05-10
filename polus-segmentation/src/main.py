from  bfio import  BioReader
import argparse, logging, sys,random
import numpy as np
from pathlib import Path
import os
import zarr
import models
from urllib.parse import urlparse
import utils

urls = [
    'https://www.cellpose.org/models/cytotorch_0',
    'https://www.cellpose.org/models/cytotorch_1',
    'https://www.cellpose.org/models/cytotorch_2',
    'https://www.cellpose.org/models/cytotorch_3',
    'https://www.cellpose.org/models/nucleitorch_0',
    'https://www.cellpose.org/models/nucleitorch_1',
    'https://www.cellpose.org/models/nucleitorch_2',
    'https://www.cellpose.org/models/nucleitorch_3',]


def download_model_weights(pretrained_model,urls=urls):
    """ Downloading model weights  based on segmentation
    Args:
        pretrained_model(str): Cyto/nuclei Segmentation
        urls(list): list of urls for model weights
    """
    # cellpose directory
    start= 0
    end=len(urls)
    if pretrained_model=='cyto':
        end+=3
    else:
        start+=4
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
            utils.download_url_to_file(url, cached_file, progress=True)

def read (inpDir,flow_path,image_names):
    """ Reads vector field and unlabelled images
    Args:
        inpDir(string): Path to unlabeled images
        flow_path(array[float32]): Path to zrr file containing vector field of labeled images
        image_names(list): list of image names
    Returns:
        image_all(list): list of unlabelled image arrays
        flow_list(list): list of vector filed for labeled images

    """

    flow_list=[]
    image_all=[]
    root = zarr.open(str(Path(flow_path).joinpath('flow.zarr')), mode='r')

    for f in image_names:
        br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
        mask_name= str(str(f).split('.',1)[0]+'.'+str(f).split('.',1)[1])

        if mask_name not in root.keys():
            logger.info('%s not present in zarr file'%mask_name)
        image_all.append(np.squeeze(br.read()))

        lbl=np.squeeze(root[mask_name]['lbl'])
        vec=np.squeeze(root[mask_name]['vector'])

        cont=np.concatenate((lbl[:,:,np.newaxis],vec), axis=2)
        cont=cont.transpose((2,0,1))

        flow_list.append(cont)

    return image_all,flow_list


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
    parser.add_argument('--flow_path',help='Flow path should be a zarr file', type=str, required=True)
    parser.add_argument('--learning_rate', required=False,
                        default=0.2, type=float, help='learning rate')
    parser.add_argument('--n_epochs', required=False,
                        default=500, type=int, help='number of epochs')
    parser.add_argument('--batch_size', required=False,
                        default=8, type=int, help='batch size')
    parser.add_argument('--residual_on', required=False,
                        default=1, type=int, help='use residual connections')
    parser.add_argument('--style_on', required=False,
                        default=1, type=int, help='use style vector')
    parser.add_argument('--concatenation', required=False,
                        default=0, type=int,
                        help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
    parser.add_argument('--train_fraction', required=False,
                        default=0.8, type=float, help='test train split')

    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    # Parse the arguments
    args = parser.parse_args() #
    diameter=args.diameter
    logger.info('diameter = {}'.format(diameter))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        inpDir = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    pretrained_model = args.pretrained_model
    logger.info('pretrained_model = {}'.format(pretrained_model))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    flow_path=args.flow_path

    train_fraction=args.train_fraction

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

    if pretrained_model is  None or not Path(cpmodel_path).exists():
        cpmodel_path = False
        logger.info('Training from scratch')
        if diameter == 0:
            rescale = False
            logger.info('Median diameter set to 0 => no rescaling during training')
        else:
            rescale = True
            szmean = diameter
    else:
        rescale = True
        diameter = szmean
        logger.info('Pretrained model  is being used')
        args.residual_on = 1
        args.style_on = 1
        args.concatenation = 0

    model = models.CellposeModel(pretrained_model=cpmodel_path,model_type=pretrained_model,diam_mean=szmean,residual_on=args.residual_on,style_on=args.style_on,concatenation=args.concatenation)
    # Surround with try/finally for proper error catching
    try:

        logger.info('Initializing ...')

        channels =[0,0]

        image_names = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes) == '.ome.tif'  ]
        inpDir_tes = [f for f in image_names if str(f).split('_')[3] == 'c1.ome.tif']
        inpDir_tes = inpDir_tes[:500]
        # Shuffle of images for test train split
        random.shuffle(image_names)
        idx = int(train_fraction * len(inpDir_tes))
        train_img_names = inpDir_tes[0:idx]
        test_img_names = inpDir_tes[idx:]
        logger.info('Running cellpose on %d train images  %d test images' %(len(train_img_names),len(test_img_names)))
        diameter = args.diameter
        logger.info('Using diameter %0.2f for all images' % diameter)

        # Read train data
        train_images,train_labels = read(inpDir,flow_path,train_img_names)

        # Read test data
        test_images,test_labels  = read(inpDir,flow_path,test_img_names)

        cpmodel_path = model.train(train_images, train_labels, train_files=train_img_names,
                                           test_data=test_images, test_labels=test_labels, test_files=test_img_names,
                                           learning_rate=args.learning_rate, channels=channels,
                                           save_path=outDir, rescale=rescale,
                                           n_epochs=args.n_epochs,
                                           batch_size=args.batch_size)
        model.pretrained_model = cpmodel_path
        logger.info('Model trained and saved to %s' % cpmodel_path)

    finally:
        logger.info('Closing the plugin')
        # Exit the program
        sys.exit()