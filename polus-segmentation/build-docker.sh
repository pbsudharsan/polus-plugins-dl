#!/bin/bash

version=$(<VERSION)
sudo docker build . -f ./Dockerfile -t labshare/polus-cellpose-training-plugin:${version}