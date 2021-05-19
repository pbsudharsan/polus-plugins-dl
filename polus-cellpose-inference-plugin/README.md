# Cellpose Inference Plugin

This plugin is an implementation of segmenting cyto/nuclei 2D/ 3D images using pretrained models
created by authors of
Cellpose. [Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1)
is a generalist algorithm for cell and nucleus segmentation. Cellpose uses two major innovations: a
reversible transformation from training set masks to vector flows that can be predicted by a neural
network, and a large segmented dataset of varied images of cells. Apart from allowing users to
specify the type of segmentation this plugin also allows the users to choose the diameter of cells as well
as the option to input custom pretrained model.

Things to keep in mind when choosing the diameter and inputting custom model:
1. Default diameter for cyto and nuclei models is 17 and 30 respectively. If 0 is passed as diameter
   plugin will estimate diameter for each image.
2. Option to estimate diameter for each image will not be available if a custom model is an input.
   However, user-specified model diameter will be used.
3. Plugin will segment cyto models by default.

This plugin saves the predicted vector field as a zarr array. This plugin has been tested with CUDA 10.1 ,
bfio:2.1.5 and runs on GPU by default.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Steps to reproduce Docker container locally

1. Pull the container

`docker pull labshare/polus-cellpose-inference-plugin:${version}`

2. Run the downloaded container

`docker run -v {inpDir}:/opt/executables/input -v {outDir}:/opt/executables/output labshare/polus-cellpose-inference-plugin:{version} --inpDir /opt/executables/input --outDir /opt/executables/output`

Add `--gpus {device no}` as an argument to use gpu in container.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents
of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 4 input argument and 1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--diameter` | Diameter| Input | number |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--pretrainedModel` | Select the model based on structure you want to segment cyto/nuclei | Input | string |
| `--cpretrainedModel` | Path to custom pretrained model | Input | string |
| `--outDir` | Output collection | Output | Generic Data type |
