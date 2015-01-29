// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ocl_wrapper.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  ocl_setup();
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::ocl_setup(){
   cl_int err=0;
   scal_kernel = clCreateKernel(amdDevice.Program, "scal_float", &err);
   diff_kernel = clCreateKernel(amdDevice.Program, "diff_float", &err);
   softmax_kernel = clCreateKernel(amdDevice.Program, "softmax_float", &err);
   d_loss = clCreateBuffer(amdDevice.Context, CL_MEM_ALLOC_HOST_PTR, sizeof(Dtype), NULL, NULL);
}

template <typename Dtype>
SoftmaxWithLossLayer<Dtype>::~SoftmaxWithLossLayer(){
  clReleaseKernel(diff_kernel);
  clReleaseKernel(scal_kernel); 
} 


template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  //the loss is computed by CPU, as GPU does poorly on reduction 
  for (int i = 0; i < num; ++i) {
    loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])],
                     Dtype(FLT_MIN)));
  }

#ifdef Track_layer
  LOG(WARNING) << "softmax with loss fp done";
#endif 
  return loss / num;
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  //return Forward_cpu(bottom, top);

  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int num = prob_.num();
  const int dim = prob_.count() / num;
  
  Dtype loss = softmax_gpu(softmax_kernel, num, dim, prob_data, label, d_loss);

#ifdef Track_layer
  LOG(WARNING) << "softmax with loss fp done";
#endif 
  return loss / num;
}


template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const Dtype* prob_data = prob_.gpu_data();
  OCL_CHECK( clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)prob_data, (cl_mem)bottom_diff, (size_t)0, (size_t)0, sizeof(Dtype) * prob_.count(), 0, NULL, NULL) );
  const Dtype* label = (*bottom)[1]->gpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  diff_gpu(diff_kernel, num, dim, bottom_diff, label);
  scal_gpu(scal_kernel, prob_.count(), Dtype(1) / num, bottom_diff);
#ifdef Track_layer
  LOG(WARNING) << "softmax with loss bp done";
#endif 
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
  }
  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
}



INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
