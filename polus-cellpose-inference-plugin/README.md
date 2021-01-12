# Cellpose 
This plugin is a implementation of   segementing images using  pretrained models  implemented by Cellpose.[Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1) is a generalist algorithm for cell and nucleus segmentation.Cellpose uses two major innovations: a reversible transformation from training set masks to vector flows that can be predicted by a neural network, and a large segmented dataset of varied images of cells. In addition, multiple smaller improvements to the basic approach led to a significant cumulative performance increase: we developed methods to use an image “style” for changing the neural network computation on an image by image basis, to validate segmented ROIs, to average the predictions of multiple models, to resize images to a common object size, to average model predictions in overlapping tiles, and to augment those predictions by flipping the tiles horizontally and vertically.

This plugin has been tested with CUDA 10.1 and can be used for  segmenting 2D and 3D images.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Steps to reproduce Docker container locally
 Pull the container
`docker pull labshare/polus-cellpose-inference-plugin:${version}` where version is the tag of the container.
 
To run this pulled container locally 
`docker run - -v {inpDir}:/opt/executables/input -v {outDir}:/opt/executables/output labshare/polus-cellpose-inference-plugin:{version} --inpDir /opt/executables/input --outDir /opt/executables/output`

By default the model will use  cyto as pretrained model.

To use gpu add argument `--gpus {device no}`


## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.


## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--diameter` | Diameter | Input | number |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--pretrained_model` | Select the model based on structure you want to segment cyto/nuclei | Input | string |
| `--cpretrained_model` | Path to custom pretrained model | Input | string |
| `--outDir` | Output collection | Output | Generic Data type |
