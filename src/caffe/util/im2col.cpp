// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
namespace caffe {


template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride - pad + h_offset;
        int w_pad = w * stride - pad + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * height * width * channels);
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride - pad + h_offset;
        int w_pad = w * stride - pad + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);


template <typename Dtype>
void im2col_gpu_ocl(cl_mem data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
    
    cl_int _err=0;
    cl_kernel Kernel = clCreateKernel(amdDevice.Program,"im2colfloat", &_err);
    if(NULL==Kernel){
        fprintf(stderr,"Failed to create kernel %d\n",_err);
    }
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_mem),(void*)&data_col);

    if(ret!=CL_SUCCESS){
        fprintf(stderr,"Failed to Set Args\n");
    }

    cl_event eventPoint;
    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {64};
    cl_int iStatus = clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,&eventPoint);
    clWaitForEvents(1,&eventPoint);
    if(CL_SUCCESS!=iStatus){
        fprintf(stderr,"Failed to enqueue kernel\n");
    }
    clReleaseKernel(Kernel);
    clReleaseEvent(eventPoint);
}

template void im2col_gpu_ocl<float>(cl_mem data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template void im2col_gpu_ocl<double>(cl_mem data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);


template <typename Dtype>
void col2im_gpu_ocl(cl_mem data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operatiors)

    cl_int _err=0;
    cl_kernel Kernel = clCreateKernel(amdDevice.Program,"col2imfloat",&_err);
    if(NULL==Kernel){
        fprintf(stderr,"Failed to create kernel %d\n",_err);
    }

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&channels);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_mem),(void*)&data_im);

    if(ret!=CL_SUCCESS){
        fprintf(stderr,"Failed to Set Args\n");
    }

    cl_event eventPoint;
    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {64};
    cl_int iStatus = clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,&eventPoint);
    clWaitForEvents(1,&eventPoint);
    if(CL_SUCCESS!=iStatus){
        fprintf(stderr,"Failed to enqueue kernel\n");
    }
    clReleaseKernel(Kernel);
    clReleaseEvent(eventPoint);
}


template void col2im_gpu_ocl<float>(cl_mem data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_gpu_ocl<double>(cl_mem data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);


}  // namespace caffe
