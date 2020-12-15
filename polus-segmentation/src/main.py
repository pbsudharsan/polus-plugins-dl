from  bfio import  BioReader

import argparse, logging, sys
import numpy as np
from pathlib import Path
import os
import zarr
import mxnet as mx
import models,utils

def read (inpDir,flow_path):
    flow_list=[]
    image_all=[]
    root = zarr.open(str(Path(flow_path).joinpath('flow.zarr')), mode='r')
    inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes) == '.ome.tif']
    for f in inpDir_files:
        br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
        if f not in root.keys():
            print('%s not present in zarr file',f)
        image_all.append(np.squeeze(br.read()))
        flow_list.append(root[f]['flow'])
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
    parser.add_argument('--unet', required=False,
                        default=0, type=int, help='run standard unet instead of cellpose flow output')
    parser.add_argument('--diameter', dest='diameter', type=float,default=30.,help='Diameter', required=False)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--pretrained_model', dest='pretrained_model', type=str,
                        help='Filename pattern used to separate data', required=False)
    parser.add_argument('--flow_path',help='Flow path should be a zarr file', type=str, required=True)
    parser.add_argument('--train_size', action='store_true', help='train size network at end of training')
    parser.add_argument('--test_dir', required=False,
                        default=[], type=str, help='folder containing test data (optional)')
    parser.add_argument('--learning_rate', required=False,
                        default=0.2, type=float, help='learning rate')
    parser.add_argument('--n_epochs', required=False,
                        default=500, type=int, help='number of epochs')
    parser.add_argument('--batch_size', required=False,
                        default=8, type=int, help='batch size')
    parser.add_argument('--residual_on', required=False,
                        default=1, type=int, help='use residual connections')
  #  parser.add_argument('--style_on', required=False,
 #                       default=1, type=int, help='use style vector')
    parser.add_argument('--concatenation', required=False,
                        default=0, type=int,
                        help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
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
    flow_path=args.flow_path
    cpmodel_path= args.pretrained_model
    if args.use_gpu:
        use_gpu = models.use_gpu()
    if use_gpu:
        device = mx.gpu()
    else:
        device = mx.cpu()
    logger.info('Using %s'%(['CPU', 'GPU'][use_gpu]))

    cpmodel_path = os.fspath(args.pretrained_model)
    szmean = 30

    if not Path(cpmodel_path).exists():
        if not args.train:
            raise ValueError('ERROR: model path missing or incorrect - cannot train size model')
        cpmodel_path = False
        print('>>>> training from scratch')
        if args.diameter == 0:
            rescale = False
            print('>>>> median diameter set to 0 => no rescaling during training')
        else:
            rescale = True
            szmean = args.diameter
    else:
        rescale = True
        args.diameter = szmean
        print('>>>> pretrained model %s is being used' % cpmodel_path)
        args.residual_on = 1
        args.style_on = 1
        args.concatenation = 0
    if rescale and args.train:
        print('>>>> during training rescaling images to fixed diameter of %0.1f pixels' % args.diameter)



    if args.unet:
        model = models.UnetModel(device=device,
                                 pretrained_model=cpmodel_path,
                                 diam_mean=szmean,
                                 residual_on=args.residual_on,
                                 style_on=args.style_on,
                                 concatenation=args.concatenation,
                                 nclasses=args.nclasses)
    else :

        model = models.CellposeModel(device=device,pretrained_model=cpmodel_path,diam_mean=szmean,residual_on=args.residual_on,style_on=args.style_on,concatenation=args.concatenation)
    # Surround with try/finally for proper error catching
    try:
        # Start the javabridge with proper java logging
        logger.info('Initializing ...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")

        # Get all file names in inpDir image collection

        channels = [args.chan, args.chan2]
        cstr0 = ['GRAY', 'RED', 'GREEN', 'BLUE']
        cstr1 = ['NONE', 'RED', 'GREEN', 'BLUE']
        logger.info('running cellpose on %d images using chan_to_seg %s and chan (opt) %s' %(len(inpDir_files), cstr0[channels[0]], cstr1[channels[1]]))
        diameter = args.diameter
        logger.info(' Using diameter %0.2f for all images' % diameter)


        try:
            if not Path(inpDir).joinpath('flow.zarr').exists():
                raise FileExistsError()


            # for m, f in root.groups():
            #     # Loop through files in inpDir image collection and process
            #     if str(f) in inpDir_files :
            #         br = BioReader(str(Path(inpDir).joinpath(f).absolute()), max_workers=1)
            #         images = np.squeeze(br.read())
            #         labels= f['flow']
            images,labels = read(inpDir,flow_path)

            cpmodel_path = model.train(images, labels, train_files=image_names,
                                           test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                           learning_rate=args.learning_rate, channels=channels,
                                           save_path=outDir, rescale=rescale,
                                           n_epochs=args.n_epochs,
                                           batch_size=args.batch_size)
            print('>>>> model trained and saved to %s' % cpmodel_path)

        except FileExistsError:
            logger.info('Zarr file does not exist. File not found in path  %r' % str((Path(inpDir).joinpath('flow.zarr'))))

    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing the plugin')
      #  jutil.kill_vm()

        # Exit the program
        sys.exit()