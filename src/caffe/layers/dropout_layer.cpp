// Copyright 2014 BVLC and contributors.

// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/ocl_util.hpp"
#include "caffe/util/ocl_wrapper.hpp"


namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.reset(new SyncedMemory(bottom[0]->count() * sizeof(int)));
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
  //if(Caffe::mode() == Caffe::GPU){
    cl_int _err;
    ocl_Kernel_Fwd = clCreateKernel(amdDevice.Program,"DropoutForwardfloat",&_err);
    ocl_Kernel_Bwd = clCreateKernel(amdDevice.Program,"DropoutBackwardfloat",&_err);
    MaskMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, bottom[0]->count()*sizeof(int), NULL, NULL);
  //} 
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
  const int count = bottom[0]->count();
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int* mask = reinterpret_cast<const int*>(rand_vec_->cpu_data());
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * mask[i] * scale_;
    }
  }
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
  const int count = bottom[0]->count();
  //int* mask_check = (int*)malloc(count * sizeof(int) );
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    OCL_CHECK( clEnqueueWriteBuffer(amdDevice.CommandQueue, MaskMem, CL_TRUE, 0, count * sizeof(int), (void*)mask, 0, NULL, NULL) );
    Dropout_fp_gpu(ocl_Kernel_Fwd, count, bottom_data, (const unsigned int*)MaskMem, (const unsigned int)uint_thres_ , (Dtype)scale_, top_data);   
    //OCL_CHECK( clEnqueueReadBuffer(amdDevice.CommandQueue, MaskMem, CL_TRUE, 0, count * sizeof(int), (void*)mask_check, 0, NULL, NULL) );
    //for(int i=0; i < count; i++)
    //if (mask[i] != mask_check[i]) 
    //LOG(INFO) << "mask not equal" << mask[i] << "vs" << mask_check[i];
  
  } else {
    OCL_CHECK( clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)bottom_data, (cl_mem)top_data, 0, 0, count*sizeof(Dtype), 0, NULL, NULL) );
  }
  //free(mask_check);
  return Dtype(0);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    //const int* mask = reinterpret_cast<const int*>(rand_vec_->cpu_data());
    const int count = (*bottom)[0]->count();
    Dropout_bp_gpu(ocl_Kernel_Bwd, count, top_diff, (const unsigned int*)MaskMem, uint_thres_ , (Dtype)scale_, bottom_diff);
  }
}


INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
