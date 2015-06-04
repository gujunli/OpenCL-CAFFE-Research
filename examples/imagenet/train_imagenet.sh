#!/usr/bin/env sh

TOOLS=../../build/tools

#GLOG_logtostderr=0 $TOOLS/train_net.bin imagenet_solver.prototxt
GLOG_logtostderr=0 $TOOLS/train_net.bin imagenet_solver_gpu.prototxt

echo "Done."
