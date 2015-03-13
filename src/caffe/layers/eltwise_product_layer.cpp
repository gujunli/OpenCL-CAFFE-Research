// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ocl_wrapper.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseProductLayer<Dtype>::ocl_setup(const int num){
     //scratch_buf = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (num*sizeof(cl_float)), NULL, NULL);
     div_kernel = clCreateKernel(amdDevice.Program, "div_float", NULL);
     mul_kernel = clCreateKernel(amdDevice.Program, "element_mul_float", NULL);
}

template <typename Dtype>
EltwiseProductLayer<Dtype>::~EltwiseProductLayer(){
     //OCL_CHECK( clReleaseMemObject(scratch_buf) );
     OCL_CHECK( clReleaseKernel(div_kernel) );
     OCL_CHECK( clReleaseKernel(mul_kernel) );
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_GE(bottom.size(), 2) <<
      "Eltwise Product Layer takes at least 2 blobs as input.";
  CHECK_EQ(top->size(), 1) <<
      "Eltwise Product Layer takes a single blob as output.";
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num, bottom[i]->num());
    CHECK_EQ(channels, bottom[i]->channels());
    CHECK_EQ(height, bottom[i]->height());
    CHECK_EQ(width, bottom[i]->width());
  }
  (*top)[0]->Reshape(num, channels, height, width);

   //initializa OpenCL kernels and cl_mem objects
   ocl_setup( num );

}

template <typename Dtype>
Dtype EltwiseProductLayer<Dtype>::Forward_cpu(
//Dtype EltwiseProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "Elt layer fp cpu";
  const int count = (*top)[0]->count();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
  for (int i = 2; i < bottom.size(); ++i) {
    caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
  }
  return Dtype(0.);
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//void EltwiseProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  LOG(INFO) << "Elt layer bp cpu";
  if (propagate_down) {
    const int count = top[0]->count();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    for (int i = 0; i < bottom->size(); ++i) {
      const Dtype* bottom_data = (*bottom)[i]->cpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
      caffe_div(count, top_data, bottom_data, bottom_diff);
      caffe_mul(count, bottom_diff, top_diff, bottom_diff);
    }
  }
}


template <typename Dtype>
Dtype EltwiseProductLayer<Dtype>::Forward_gpu(
//Dtype EltwiseProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  //LOG(INFO) << "EltwiseProductLAyer fp_GPU";
  const int count = (*top)[0]->count();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  caffe_gpu_mul(mul_kernel, count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
  for (int i = 2; i < bottom.size(); ++i) {
    caffe_gpu_mul(mul_kernel, count, top_data, bottom[i]->gpu_data(), top_data);
  }
  return Dtype(0.);
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//void EltwiseProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  //LOG(INFO) << "EltwiseProductLAyer bp_GPU";
  if (propagate_down) {
    const int count = top[0]->count();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    for (int i = 0; i < bottom->size(); ++i) {
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      bool initialized = false;
      for (int j = 0; j < bottom->size(); ++j) {
        if (i == j) { continue; }
        if (!initialized) {
            caffe_gpu_copy(count, (*bottom)[j]->gpu_data(), bottom_diff);
            initialized = true;
        } else {
            caffe_gpu_mul(count, (*bottom)[j]->gpu_data(), bottom_diff,
                        bottom_diff);
        }
      }
      //caffe_gpu_div(div_kernel, count, top_data, bottom_data, bottom_diff);
      caffe_gpu_mul(mul_kernel, count, bottom_diff, top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(EltwiseProductLayer);


}  // namespace caffe
