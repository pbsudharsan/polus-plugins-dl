#!/bin/bash

version=$(<VERSION)
 docker build . -f ./Dockerfile-gpu -t labshare/polus-cellpose-inference-plugin:${version}