// Copyright 2014 AMD DNN contributors.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include "caffe/common.hpp"
#include "caffe/util/ocl_util.hpp"
namespace caffe {


template <typename Dtype>
void get_max_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* bottom_data, Dtype* scale_data){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&dim) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&bottom_data) );
    OCL_CHECK( clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&scale_data) );
 
    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void get_max_gpu<float>(cl_kernel Kernel, const int num, const int dim, const float* bottom_data, float* scale_data);
template void get_max_gpu<double>(cl_kernel Kernel, const int num, const int dim, const double* bottom_data, double* scale_data);


template <typename Dtype>
void exp_gpu(cl_kernel Kernel, const int num, const Dtype* data, Dtype* out){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&data) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&out) );

    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void exp_gpu<float>(cl_kernel Kernel, const int num, const float* data, float* out);
template void exp_gpu<double>(cl_kernel Kernel, const int num, const double* data, double* out);

template <typename Dtype>
void softmax_div_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* scale, Dtype* data){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&dim) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&scale) );
    OCL_CHECK( clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&data) );

    size_t Global_Work_Size[1] = {num*dim};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void softmax_div_gpu<float>(cl_kernel Kernel, const int num, const int dim, const float* scale, float* data);
template void softmax_div_gpu<double>(cl_kernel Kernel, const int num, const int dim, const double* scale, double* data);

template <typename Dtype>
void scal_gpu(cl_kernel Kernel, const int num, const Dtype alpha, Dtype* data){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*)&alpha) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&data) );

    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void scal_gpu<float>(cl_kernel Kernel, const int num, const float alpha, float* data);
template void scal_gpu<double>(cl_kernel Kernel, const int num, const double alpha, double* data);

template <typename Dtype>
void diff_gpu(cl_kernel Kernel, const int num, int dim, Dtype* data, const Dtype* label){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&dim) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&data) );
    OCL_CHECK( clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&label) );

    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void diff_gpu<float>(cl_kernel Kernel, const int num, const int dim, float* data, const float* label);
template void diff_gpu<double>(cl_kernel Kernel, const int num, const int dim, double* data, const double* label);

template <typename Dtype>
void max_pool_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, Dtype* top_data){
	cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*)&clnum);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*)&channels_);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*)&height_);
    ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*)&width_);
    ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*)&stride_);
    ret |= clSetKernelArg(Kernel,10, sizeof(cl_mem), (void*)&top_data);
    OCL_CHECK(ret);

    size_t Global_Work_Size[] = {count * 1};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template  void max_pool_fp_gpu<float>(cl_kernel Kernel, const int count, const float* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, float* top_data);
template  void max_pool_fp_gpu<double>(cl_kernel Kernel, const int count, const double* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, double* top_data);

template <typename Dtype> 
void ave_pool_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, Dtype* top_data){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*)&clnum);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*)&channels_);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*)&height_);
    ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*)&width_);
    ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*)&stride_);
    ret |= clSetKernelArg(Kernel, 10,sizeof(cl_int), (void*)&pad_);
    ret |= clSetKernelArg(Kernel, 11,sizeof(cl_mem), (void*)&top_data);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {count * 1};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void ave_pool_fp_gpu<float>(cl_kernel Kernel, const int count, const float* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, float* top_data);
template void ave_pool_fp_gpu<double>(cl_kernel Kernel, const int count, const double* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_,const int stride_,const int pad_, double* top_data);

template <typename Dtype> 
void max_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const Dtype* top_data, const Dtype* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, Dtype* bottom_diff ){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&top_data);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&top_diff);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*)&clnum);
    ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*)&channels_);
    ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*)&height_);
    ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*)&width_);
    ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel,10, sizeof(cl_int), (void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel,11, sizeof(cl_int), (void*)&stride_);
    ret |= clSetKernelArg(Kernel,12, sizeof(cl_mem), (void*)&bottom_diff);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {count};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void max_pool_bp_gpu<float>(cl_kernel Kernel, const int count, const float* bottom_data, const float* top_data, const float* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, float* bottom_diff);
template void max_pool_bp_gpu<double>(cl_kernel Kernel, const int count, const double* bottom_data, const double* top_data, const double* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, double* bottom_diff );

template <typename Dtype> 
void ave_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, Dtype* bottom_diff){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&top_diff);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*)&clnum);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*)&channels_);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*)&height_);
    ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*)&width_);
    ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*)&stride_);
    ret |= clSetKernelArg(Kernel,10, sizeof(cl_int), (void*)&pad_);
    ret |= clSetKernelArg(Kernel,11,sizeof(cl_mem),(void*)&bottom_diff);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[]={count};
    size_t uiLocal_Work_Size[]={256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,NULL));
}

template void ave_pool_bp_gpu<float>(cl_kernel Kernel, const int count, const float* top_diff, const int clnum, const int channels_, const int intheight_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, float* bottom_diff);
template void ave_pool_bp_gpu<double>(cl_kernel Kernel, const int count, const double* top_diff, const int clnum, const int channels_, const int intheight_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, double* bottom_diff);

template <typename Dtype> 
void Relu_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, Dtype* top_data){
     cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&top_data);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {count * 1};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void Relu_fp_gpu<float>(cl_kernel Kernel, const int count, const float* bottom_data, float* top_data);
template void Relu_fp_gpu<double>(cl_kernel Kernel, const int count, const double* bottom_data, double* top_data);

template <typename Dtype> 
void Relu_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&top_diff);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&bottom_diff);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {count * 1};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void Relu_bp_gpu<float>(cl_kernel Kernel, const int count, const float* top_diff, const float* bottom_data, float* bottom_diff);
template void Relu_bp_gpu<double>(cl_kernel Kernel, const int count, const double* top_diff, const double* bottom_data, double* bottom_diff);
}  // namespace caffe
