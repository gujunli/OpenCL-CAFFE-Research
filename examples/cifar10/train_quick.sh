#!/usr/bin/env sh

TOOLS=../../build/tools

#LOG(INFO) << "file"
#   FLAGS_logtostderr = 0
#  FLAGS_log_dir = "/home/jlgu/Documents"

GLOG_logtostderr=1 $TOOLS/train_net.bin cifar10_quick_solver.prototxt
#$TOOLS/train_net.bin cifar10_quick_solver.prototxt


#reduce learning rate by fctor of 10 after 8 epochs
#GLOG_logtostderr=1 $TOOLS/train_net.bin cifar10_quick_solver_lr1.prototxt cifar10_quick_iter_4000.solverstate
