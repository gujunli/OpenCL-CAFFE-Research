// Copyright 2014 BVLC and contributors.

#include <CL/cl.h>
#include <cstring>
#include <stdio.h>
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/ocl_util.hpp"


namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

  if (gpu_ptr_) {
    //CUDA_CHECK(cudaFree(gpu_ptr_));
    clReleaseMemObject((cl_mem)gpu_ptr_);
  }
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
    memset(cpu_ptr_, 0, size_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:{
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    OCL_CHECK(clEnqueueReadBuffer(amdDevice.CommandQueue, (cl_mem)gpu_ptr_, CL_TRUE, 0, size_, cpu_ptr_, 0, NULL, NULL));
    head_ = SYNCED;
#ifdef Track_data_transfer
    LOG(WARNING) << "sync data from GPU to CPU";
#endif
    break;
  }
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  switch (head_) {
  case UNINITIALIZED:{
    //CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    //To Do: implement OCL_CHECK_NULL
    cl_mem tmpMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, size_, NULL, NULL);
    if(NULL == tmpMem){
      fprintf(stderr,"Failed to create memory object 58\n");
      break;
    }
    ocl_memset(tmpMem, (int)0, (int)(size_/sizeof(int)));

    gpu_ptr_ = (void*)tmpMem; 
    head_ = HEAD_AT_GPU;
    break;
  }
  case HEAD_AT_CPU:{
    if (gpu_ptr_ == NULL) {
      //CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      cl_mem tmpMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, size_, NULL, NULL);
      if(NULL == tmpMem){
        fprintf(stderr,"Failed to create memory object\n");
      }
      gpu_ptr_ = (void*)tmpMem;
    }
    OCL_CHECK(clEnqueueWriteBuffer(amdDevice.CommandQueue, (cl_mem)gpu_ptr_, CL_TRUE, 0, size_, cpu_ptr_, 0, NULL, NULL));
    head_ = SYNCED;
#ifdef Track_data_transfer
    LOG(WARNING) << "sync data from CPU to GPU";
#endif
    break;
  }
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  to_gpu();
  return (const void*)gpu_ptr_;
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
}


}  // namespace caffe

