// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
Dtype ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
    cl_int _err=0;
    cl_kernel Kernel = clCreateKernel(amdDevice.Program,"ReLUForwardfloat",&_err);
    if(NULL == Kernel){
        fprintf(stderr,"Failed to create kernel %d\n",_err);
    }
    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&count);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&bottom_data);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_mem),(void*)&top_data);
    OCL_CHECK(ret);
    cl_event eventPoint;
    size_t Global_Work_Size[]={count * 1};
    size_t Local_Work_Size[]={256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL, Global_Work_Size, Local_Work_Size,0,NULL,&eventPoint));
    clWaitForEvents(1,&eventPoint);
    clReleaseKernel(Kernel);
    clReleaseEvent(eventPoint);
  return Dtype(0);
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();

    cl_int _err=0;
    cl_kernel Kernel = clCreateKernel(amdDevice.Program,"ReLUBackwardfloat",&_err);
    if(NULL == Kernel){
        fprintf(stderr,"Failed to create kernel %d\n",_err);
    }
    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&count);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&top_diff);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_mem),(void*)&bottom_data);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_mem),(void*)&bottom_diff);
    OCL_CHECK(ret);
    cl_event eventPoint;
    size_t Global_Work_Size[]={count * 1};
    size_t Local_Work_Size[]={256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL, Global_Work_Size, Local_Work_Size,0,NULL,&eventPoint));
    clWaitForEvents(1,&eventPoint);
    clReleaseKernel(Kernel);
    clReleaseEvent(eventPoint);

  }
}


INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
