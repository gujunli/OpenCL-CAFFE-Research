// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/ocl_wrapper.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {
template <typename Dtype>
void ReLULayer<Dtype>::ocl_setup(){
    cl_int _err=0;
    ReLUForward_kernel = clCreateKernel(amdDevice.Program,"ReLUForwardfloat",&_err);
    ReLUBackward_kernel = clCreateKernel(amdDevice.Program,"ReLUBackwardfloat",&_err);
}

template <typename Dtype>
ReLULayer<Dtype>::~ReLULayer(){
  OCL_CHECK( clReleaseKernel(ReLUForward_kernel) );
  OCL_CHECK( clReleaseKernel(ReLUBackward_kernel) );
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Relu_fp_gpu(ReLUForward_kernel,count,bottom_data,top_data);
#ifdef Track_layer
    LOG(WARNING) << "ReLu fp done";
#endif
  return Dtype(0);
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = (*top)[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
        top_data[i] = max(bottom_data[i], Dtype(0));
    }
    return Dtype(0);
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                    const bool propagate_down,
                                    vector<Blob<Dtype>*>* bottom) {
    if (propagate_down) {
        const Dtype* bottom_data = (*bottom)[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
        const int count = (*bottom)[0]->count();
        for (int i = 0; i < count; ++i) {
            bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
        }
    }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    Relu_bp_gpu(ReLUBackward_kernel,count,top_diff,bottom_data,bottom_diff);
#ifdef Track_layer
    LOG(WARNING) << "ReLu bp done";
#endif
  }
}

INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
