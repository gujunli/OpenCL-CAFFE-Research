// Copyright 2014 BVLC and contributors.

// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"

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
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    
    cl_int status;
    status = clEnqueueWriteBuffer(amdDevice.CommandQueue, MaskMem, CL_TRUE, 0, count * sizeof(int), (void*)mask, 0, NULL, NULL);
    if(status != CL_SUCCESS){
      fprintf(stderr,"Failed to write buffer\n");
    }

    // DropoutForward(cl_kernel kernel, count, bottom_data, MaskMem, uint_thres_, scale_, top_data);   
    cl_int ret;
    ret=clSetKernelArg(ocl_Kernel_Fwd,0,sizeof(cl_int),(void*)&count);
    ret|=clSetKernelArg(ocl_Kernel_Fwd,1,sizeof(cl_mem),(void*)&bottom_data);
    ret|=clSetKernelArg(ocl_Kernel_Fwd,2,sizeof(cl_mem),(void*)&MaskMem);
    ret|=clSetKernelArg(ocl_Kernel_Fwd,3,sizeof(cl_int),(void*)&threshold_); 
    ret|=clSetKernelArg(ocl_Kernel_Fwd,4,sizeof(cl_float),(void*)&scale_); 
    ret|=clSetKernelArg(ocl_Kernel_Fwd,5,sizeof(cl_mem),(void*)&top_data); 

    if(ret!=CL_SUCCESS){
      fprintf(stderr,"Failed to Set Args\n");
    }   
 
    cl_event eventPoint;
    size_t uiGlobal_Work_Size[]={count};
    size_t uiLocal_Work_Size[]={256};
    cl_int iStatus = clEnqueueNDRangeKernel(amdDevice.CommandQueue,ocl_Kernel_Fwd, 1, NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,&eventPoint);
    if(CL_SUCCESS!=iStatus){
      fprintf(stderr,"Failed to enqueue kernel\n");
    }
  } else {
    cl_int status;
    status = clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)bottom_data, (cl_mem)top_data, 0, 0, count*sizeof(Dtype), 0, NULL, NULL);
    if(status != CL_SUCCESS){
      fprintf(stderr,"Failed to Copy buffer\n");
    }   
  }
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
    
    //DropoutBackward(const int n, const Dtype* in_diff, const unsigned int* mask, const unsigned int threshold, const float scale,Dtype* out_diff)

    cl_int ret;
    ret=clSetKernelArg(ocl_Kernel_Bwd,0,sizeof(cl_int),(void*)&count);
    ret|=clSetKernelArg(ocl_Kernel_Bwd,1,sizeof(cl_mem),(void*)&top_diff);
    ret|=clSetKernelArg(ocl_Kernel_Bwd,2,sizeof(cl_mem),(void*)&MaskMem);
    ret|=clSetKernelArg(ocl_Kernel_Bwd,3,sizeof(cl_int),(void*)&threshold_); 
    ret|=clSetKernelArg(ocl_Kernel_Bwd,4,sizeof(cl_float),(void*)&scale_); 
    ret|=clSetKernelArg(ocl_Kernel_Bwd,5,sizeof(cl_mem),(void*)&bottom_diff); 
   
    cl_event eventPoint;
    size_t uiGlobal_Work_Size[]={count};
    size_t uiLocal_Work_Size[]={64};
    cl_int iStatus = clEnqueueNDRangeKernel(amdDevice.CommandQueue,ocl_Kernel_Bwd, 1, NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,&eventPoint);
    if(CL_SUCCESS!=iStatus){
      fprintf(stderr,"Failed to enqueue kernel\n");
    }
  }
}


INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
