// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Im2col Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Im2col Layer takes a single blob as output.";
  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  stride_ = this->layer_param_.convolution_param().stride();
  pad_ = this->layer_param_.convolution_param().pad();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  (*top)[0]->Reshape(bottom[0]->num(), channels_ * kernel_size_ * kernel_size_,
      (height_ + 2 * pad_ - kernel_size_) / stride_ + 1,
      (width_ + 2 * pad_ - kernel_size_) / stride_ + 1);
   //OpenCL related initialization
  ocl_setup(bottom[0]->offset(1)*sizeof(Dtype), (*top)[0]->offset(1)*sizeof(Dtype));
}

template <typename Dtype>
void Im2colLayer<Dtype>::ocl_setup(const int bottom0_offset1, const int top0_offset1){
    col_data = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)top0_offset1, NULL, NULL);
    sub_bottom = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)bottom0_offset1, NULL, NULL);
    sub_top_diff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)bottom0_offset1, NULL, NULL);
    sub_bottom_diff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)top0_offset1, NULL, NULL);

    im2col_kernel = clCreateKernel(amdDevice.Program,"im2colfloat", NULL);
    col2im_kernel = clCreateKernel(amdDevice.Program,"col2imfloat", NULL);
}

template <typename Dtype>
Im2colLayer<Dtype>::~Im2colLayer(){
  OCL_CHECK( clReleaseMemObject(col_data) );
  OCL_CHECK( clReleaseMemObject(sub_bottom) );
  OCL_CHECK( clReleaseMemObject(sub_top_diff) );
  OCL_CHECK( clReleaseMemObject(sub_bottom_diff) );
  OCL_CHECK( clReleaseKernel(im2col_kernel) );
  OCL_CHECK( clReleaseKernel(col2im_kernel) );
}

template <typename Dtype>
Dtype Im2colLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
        width_, kernel_size_, pad_, stride_, top_data + (*top)[0]->offset(n));
  }
  return Dtype(0.);
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    col2im_cpu(top_diff + top[0]->offset(n), channels_, height_, width_,
        kernel_size_, pad_, stride_, bottom_diff + (*bottom)[0]->offset(n));
  }
}

template <typename Dtype>
Dtype Im2colLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
   OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)bottom_data, (cl_mem)sub_bottom, (size_t)(bottom[0]->offset(n)*sizeof(Dtype)), 0, bottom[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
    im2col_gpu(im2col_kernel,sub_bottom, channels_, height_,
        width_, kernel_size_, pad_, stride_, (Dtype*)col_data);
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)col_data, (cl_mem)top_data, (size_t)((*top)[0]->offset(n)*sizeof(Dtype)), 0, (*top)[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
  }
  return Dtype(0.);
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)top_diff,  (cl_mem)sub_top_diff, (size_t)(top[0]->offset(n)*sizeof(Dtype)), 0, top[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
    col2im_gpu(col2im_kernel, sub_top_diff, channels_, height_, width_,
        kernel_size_, pad_, stride_, (Dtype*)sub_bottom_diff);
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)sub_bottom_diff,  (cl_mem)bottom_diff, (size_t)((*bottom)[0]->offset(n)*sizeof(Dtype)), 0, (*bottom)[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
  }
}

INSTANTIATE_CLASS(Im2colLayer);

}  // namespace caffe
