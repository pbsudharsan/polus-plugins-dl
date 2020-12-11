#!/bin/bash

version=$(<VERSION)
 sudo docker build . -f ./Dockerfile-gpu -t labshare/polus-cellpose-inference-plugin:${version}