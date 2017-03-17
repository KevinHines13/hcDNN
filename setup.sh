#! /bin/sh

BUILD_DIR=build
CUDNN_PATH=${HOME}/cudnn
if [ ! -d ${BUILD_DIR} ]; then
  mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

CXX=hcc cmake -DCMAKE_BUILD_TYPE=debug -DCOMPILE_HCC=On ..
