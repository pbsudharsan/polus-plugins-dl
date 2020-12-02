# Cellpose 

 Cellpose , https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Features to be added
   Will provide option to input model weights. 
   Support for colour images.
   Seperate diameter prediction and probability prediction into different plugins

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| '`--use_gpu`   |  Running on Gpu         | Input | number |
| `--diameter` | Diameter | Input | number |
|`--chan `    | choose the channel to segment|Input | string |
|`--chan2 `   | choose the 2nd  channel to  segment|Input | string |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--pretrained_model` | Filename pattern used to separate data | Input | string |
| `--outDir` | Output collection | Output | collection |

