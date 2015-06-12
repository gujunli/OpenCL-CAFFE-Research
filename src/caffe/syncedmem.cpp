// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <stdio.h>
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/ocl_util.hpp"

#define CL_MEM_USE_PERSISTENT_MEM_AMD       (1 << 6)        // Alloc from GPU's CPU visible heap

namespace caffe {

long long unsigned device_mem_consumption = 0;

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    OCL_CHECK( clEnqueueUnmapMemObject(amdDevice.CommandQueue, (cl_mem)gpu_cache_ptr_, cpu_ptr_, 0, NULL, NULL) );
    clFinish(amdDevice.CommandQueue);
  }
  if(gpu_cache_ptr_ && own_cpu_data_)  {
    OCL_CHECK( clReleaseMemObject((cl_mem)gpu_cache_ptr_) );
  }
  if (gpu_ptr_) {
    OCL_CHECK( clReleaseMemObject((cl_mem)gpu_ptr_) );
  }

  clReleaseKernel(oclmem_kernel);
}

void SyncedMemory::ocl_setup() {
  cl_int err=0;
  oclmem_kernel = clCreateKernel(amdDevice.Program, "OCL_memset2", &err);
  OCL_CHECK(err);
}

inline void SyncedMemory::to_cpu() {

  switch (head_) {
  case UNINITIALIZED:
    //allocate pre-pinned memory
    //pinned_buffer_ptr_
    if(data_layer_){
      gpu_cache_ptr_ = clCreateBuffer(amdDevice.Context, CL_MEM_USE_PERSISTENT_MEM_AMD, size_, NULL, NULL);
    //
#ifdef  print_memory_trace
      device_mem_consumption += size_;
      printf("device_mem_consumption = %lu, total device_mem_consumption = %lu\n", size_/4, device_mem_consumption);
#endif
    }
    else{
      gpu_cache_ptr_ = clCreateBuffer(amdDevice.Context, CL_MEM_ALLOC_HOST_PTR, size_, NULL, NULL);
    }
    cpu_ptr_ = clEnqueueMapBuffer(amdDevice.CommandQueue, (cl_mem)gpu_cache_ptr_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size_, 0, NULL, NULL, NULL);
    memset(cpu_ptr_, 0, size_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:{
    if (cpu_ptr_ == NULL) {
      gpu_cache_ptr_ = clCreateBuffer(amdDevice.Context, CL_MEM_ALLOC_HOST_PTR, size_, NULL, NULL);
      cpu_ptr_ = clEnqueueMapBuffer(amdDevice.CommandQueue, (cl_mem)gpu_cache_ptr_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size_, 0, NULL, NULL, NULL);
      own_cpu_data_ = true;
    }
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)gpu_ptr_, (cl_mem)gpu_cache_ptr_, 0, 0, size_, 0, NULL, NULL));
    clFinish(amdDevice.CommandQueue);
    head_ = SYNCED;
#ifdef Track_data_transfer
    LOG(WARNING) << "sync: data from GPU to CPU";
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
    //To Do: implement OCL_CHECK_NULL
    cl_mem tmpMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, size_, NULL, NULL);
   // 
#ifdef  print_memory_trace
    device_mem_consumption += size_;
    printf("device_mem_consumption = %lu, total device_mem_consumption = %lu\n", size_/4, device_mem_consumption);
#endif 
   if(NULL == tmpMem){
      fprintf(stderr,"Failed to create memory object 58\n");
      break;
    }
    ocl_memset(oclmem_kernel, tmpMem, (int)0, (int)(size_/sizeof(int)));

    gpu_ptr_ = (void*)tmpMem; 
    head_ = HEAD_AT_GPU;
    break;
  }
  case HEAD_AT_CPU:{
    if (gpu_ptr_ == NULL) {
      cl_mem tmpMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, size_, NULL, NULL);
     //
#ifdef  print_memory_trace
      device_mem_consumption += size_;
      printf("device_mem_consumption = %lu, total device_mem_consumption = %lu\n", size_/4, device_mem_consumption);
#endif
      if(NULL == tmpMem){
        fprintf(stderr,"Failed to create memory object\n");
      }
      gpu_ptr_ = (void*)tmpMem;
    }

    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)gpu_cache_ptr_, (cl_mem)gpu_ptr_, 0, 0, size_, 0, NULL, NULL));
    clFinish(amdDevice.CommandQueue);
    head_ = SYNCED;
#ifdef Track_data_transfer
    LOG(WARNING) << "sync: data from CPU to GPU";
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
  OCL_CHECK( clEnqueueUnmapMemObject(amdDevice.CommandQueue, (cl_mem) gpu_cache_ptr_, cpu_ptr_, 0, NULL, NULL));
  OCL_CHECK( clReleaseMemObject((cl_mem) gpu_cache_ptr_));
  clFinish(amdDevice.CommandQueue); //is this necessary?
  }
  gpu_cache_ptr_ = clCreateBuffer(amdDevice.Context, CL_MEM_USE_HOST_PTR, size_, data, NULL);
  cpu_ptr_ = clEnqueueMapBuffer(amdDevice.CommandQueue, (cl_mem)gpu_cache_ptr_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size_, 0, NULL, NULL, NULL);
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}


//TO_DO Junli: implement set_gpu_data(prefetech_data_ptr)
//  {gpu_ptr = prefetch_data_ptr; head_ = HEAD_AT_GPU; }
const void* SyncedMemory::gpu_data() {
  to_gpu();
  return (const void*)gpu_ptr_;
}

const void* SyncedMemory::gpu_cache_data(){
  to_cpu();
  return (const void*)gpu_cache_ptr_;
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

