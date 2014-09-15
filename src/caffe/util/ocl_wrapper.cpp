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
void get_max_gpu( const int num, const int dim, const Dtype* bottom_data, Dtype* scale_data){
    cl_int err=0;
    cl_kernel Kernel;
    //if (Dtype == "float")
    	Kernel = clCreateKernel(amdDevice.Program, "get_max_float", &err);
    //if (Dtype == "double")
    //	Kernel = clCreateKernel(amdDevice.Program, "get_max_doulbe", &err);
    if(NULL==Kernel){
        fprintf(stderr, "Failed to create kernel %d\n", err);
    }   
 
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&dim) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&bottom_data) );
    OCL_CHECK( clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&scale_data) );
 
    cl_event event;
    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, &event) );
    OCL_CHECK( clWaitForEvents(1, &event));
    
    clReleaseKernel(Kernel);
    clReleaseEvent(event);

}

// Explicit instantiation
template void get_max_gpu<float>(const int num, const int dim, const float* bottom_data, float* scale_data);
template void get_max_gpu<double>(const int num, const int dim, const double* bottom_data, double* scale_data);


template <typename Dtype>
void exp_gpu( const int num, const Dtype* data, Dtype* out){
    cl_int err=0;
    cl_kernel Kernel;
    //if (Dtype == 'float')
    	Kernel = clCreateKernel(amdDevice.Program, "exp_float", &err);
    //if (Dtype == 'double')
    //	Kernel = clCreateKernel(amdDevice.Program, "exp_doulbe", &err);
    if(NULL==Kernel){
        fprintf(stderr, "Failed to create kernel %d\n", err);
    }

    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&data) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&out) );

    cl_event event;
    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, &event) );
    OCL_CHECK( clWaitForEvents(1, &event));

    clReleaseKernel(Kernel);
    clReleaseEvent(event);

}

// Explicit instantiation
template void exp_gpu<float>(const int num, const float* data, float* out);
template void exp_gpu<double>(const int num, const double* data, double* out);



template <typename Dtype>
void softmax_div_gpu( const int num, const int dim, const Dtype* scale, Dtype* data){
    cl_int err=0;
    cl_kernel Kernel;
    //if (Dtype == 'float')
        Kernel = clCreateKernel(amdDevice.Program, "softmax_div_float", &err);
    //if (Dtype == 'double')
      //  Kernel = clCreateKernel(amdDevice.Program, "softmax_div_doulbe", &err);
    if(NULL==Kernel){
        fprintf(stderr, "Failed to create kernel %d\n", err);
    }

    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&dim) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&scale) );
    OCL_CHECK( clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&data) );

    cl_event event;
    size_t Global_Work_Size[1] = {num*dim};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, &event) );
    OCL_CHECK( clWaitForEvents(1, &event));

    clReleaseKernel(Kernel);
    clReleaseEvent(event);

}

// Explicit instantiation
template void softmax_div_gpu<float>(const int num, const int dim, const float* scale, float* data);
template void softmax_div_gpu<double>(const int num, const int dim, const double* scale, double* data);


}  // namespace caffe
