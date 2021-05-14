# Cellpose Training plugin

This plugin lets users Cellpose models to train a model to segment cyto/nuclei in 2d images.
Cellpose is a generalist algorithm for cell and nucleus segmentation.

Plugin requires user to specify:

1. Path to zarr file containing vector field and labels of the images.
2. Path to unlabelled image collection.

All the other arguments are default parameters but have default value. Check options for the default
value.

Things to keep in mind when choosing values for the arguments:

1. If 0 is passed as argument for diameter no rescaling of images is done.
2. Choosing cyto/nuclei as arguments for pretrainedModel makes the plugin use weights released by
   authors of the repo. Ignore the argument to train from scratch. This argument can also be used to
   input custom pretrained models.
3. Training will stop if test loss hasn't reduced over 5 computations i.e. Early stopping is set to 5.

Check options for default argument values if there are any. This plugin outputs weights after
training. Code has been tested with CUDA 10.1 , bfio:2.0.4 and runs on GPU by default.

For more information on the neural network
visit  [Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1). Check out their repo
for [mxnet implementation](https://github.com/MouseLand/cellpose/tree/master/cellpose).

## Building

To build the Docker image for the conversion plugin, run

`./build-docker.sh`.

## Steps to reproduce Docker container locally

1. Pull the container

`docker pull labshare/polus-cellpose-training-plugin:${version}`

2. Run the downloaded container

`docker run -v {inpDir}:/opt/executables/input -v {zarr file}}:/opt/executables/flow  -v {outDir}:/opt/executables/output labshare/polus-cellpose-training-plugin:{version} --inpDir /opt/executables/input --flowPath  /opt/executables/flow --outDir /opt/executables/output`

Add `--gpus {device no}` as an argument to use gpu in container.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents
of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 11 input argument and 1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
|`--diameter` | Diameter(default cyto-30 nuclei-17) | Input | number |
|`--flowPath` | Labelled images as a vector field | Input | GenericData |
|`--learningRate` | Learning rate(default 0.2) | Input | number |
|`--nEpochs` | Number of epochs(default 500) | Input | number |
|`--batchSize` | Batch size(default 8) |Input | number |
|`--residualOn` | Use residual connections(default True ) | Input | boolean |
|`--styleOn` | Use style vector(default True ) | Input | boolean |
|`--concatenation` | Concatenate downsampled layers with upsampled layers(default False ) | Input | boolean |
| `--inpDir` | Input Unlabelled image collection to be processed by this plugin | Input | collection |
| `--pretrainedModel` | Select the model based on structure you want to segment cyto/nuclei/custom model path | Input | string |
| `--cpretrainedModel` | Path to custom pretrained model | Input | string |
| `--outDir` | Output collection | Output | Tensorflow model |










