// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ocl_util.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Conv Layer takes a single blob as output.";
  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  stride_ = this->layer_param_.convolution_param().stride();
  group_ = this->layer_param_.convolution_param().group();
  pad_ = this->layer_param_.convolution_param().pad();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  CHECK_EQ(channels_ % group_, 0);
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int width_out = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  col_buffer_.Reshape(
      1, channels_ * kernel_size_ * kernel_size_, height_out, width_out);
  // Set the parameters
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_size_ * kernel_size_ / group_;
  N_ = height_out * width_out;
  (*top)[0]->Reshape(bottom[0]->num(), num_output_, height_out, width_out);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(
        num_output_, channels_ / group_, kernel_size_, kernel_size_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Set up the bias filler
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
}


template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  cl_mem subTopMem=clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(top_offset*sizeof(Dtype)), NULL, NULL);
  cl_mem subWgtMem=clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(weight_offset*sizeof(Dtype)), NULL, NULL);
  cl_mem subBotMem=clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(bottom[0]->offset(1))*sizeof(Dtype), NULL, NULL);
  cl_mem subColMem=clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(col_offset)*sizeof(Dtype), NULL, NULL);
  for (int n = 0; n < num_; ++n) {
    // First, im2col
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)bottom_data, (cl_mem)subBotMem, (size_t)(bottom[0]->offset(n)*sizeof(Dtype)), 0, bottom[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
    
    im2col_gpu_ocl((cl_mem)subBotMem, channels_, height_, 
                       width_, kernel_size_, pad_, stride_, col_data);
    //note if(group!=1) it doesn't work
    for (int g = 0; g < group_; ++g) {
      OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)col_data, (cl_mem)subColMem, (size_t)(col_offset * g * sizeof(Dtype)), 0, col_offset *sizeof(Dtype), 0, NULL, NULL));

      OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)weight, (cl_mem)subWgtMem, (size_t)(weight_offset * g * sizeof(Dtype)), 0, weight_offset * sizeof(Dtype), 0, NULL, NULL));

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., (Dtype*)subWgtMem, (Dtype*)subColMem,
        (Dtype)0., (Dtype*)subTopMem);
    }

    // third, add bias
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          (Dtype)1., (Dtype*)subTopMem);
    }
    
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)subTopMem, (cl_mem)top_data, 0, (size_t)((*top)[0]->offset(n) * sizeof(Dtype)), ((*top)[0]->count()/num_)*sizeof(Dtype), 0, NULL, NULL));
  }
#ifdef Track_layer
  LOG(WARNING) << "conv fp done";
#endif
  clReleaseMemObject(subTopMem);
  clReleaseMemObject(subWgtMem);
  clReleaseMemObject(subBotMem);
  clReleaseMemObject(subColMem);
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  
  const Dtype* top_diff_ = top[0]->gpu_diff();
  const Dtype* weight_ = this->blobs_[0]->gpu_data();
  Dtype* weight_diff_ = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data_ = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff_ = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data_ = col_buffer_.mutable_gpu_data();
  Dtype* col_diff_ = col_buffer_.mutable_gpu_diff();
  
  Dtype* bias_diff = NULL;
  Dtype* bias_diff_ = NULL;
  cl_mem sub_diff_ = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(top[0]->offset(1)*sizeof(Dtype)), NULL, NULL);
  cl_mem sub_bottom_ = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)((*bottom)[0]->offset(1)*sizeof(Dtype)), NULL, NULL);
  cl_mem sub_tmp_ = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)((*bottom)[0]->offset(1)*sizeof(Dtype)), NULL, NULL);

  if (bias_term_) {
    bias_diff_ = this->blobs_[1]->mutable_gpu_diff();
    ocl_memset(bias_diff_, (Dtype)(0.), this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemvv<Dtype>(CblasNoTrans, M_, N_,
          (Dtype)1., top_diff_, top[0]->offset(n), N_,
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), (size_t)0, (Dtype)1., 1,
          bias_diff_, (size_t)0, 1);
    }
  }
  
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  cl_int err;
  
  for (int n = 0; n < num_; ++n) {
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)top_diff_, sub_diff_, (size_t)(top[0]->offset(n)*sizeof(Dtype)), 0, (top[0]->offset(1)*sizeof(Dtype)), 0, NULL, NULL));
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)bottom_data_, sub_bottom_, (size_t)((*bottom)[0]->offset(n)*sizeof(Dtype)), 0, (*bottom)[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_gpu_ocl(sub_bottom_, channels_, height_,
                      width_, kernel_size_, pad_, stride_, col_data_);
   
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., (Dtype*)sub_diff_,
        (Dtype*)col_data_, (Dtype)1.,
        (Dtype*)weight_diff_);
    }

   if (propagate_down) {
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight_ ,
          (Dtype*)sub_diff_,
          (Dtype)0., col_diff_);
      }
    
      // col2im back to the data
      col2im_gpu_ocl((cl_mem)col_diff_, channels_, height_, width_, kernel_size_, pad_,
          stride_, (Dtype*)sub_tmp_);
      OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, sub_tmp_, (cl_mem)bottom_diff_, 0, (size_t)((*bottom)[0]->offset(n)*sizeof(Dtype)), (*bottom)[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
    }

  }
    clReleaseMemObject(sub_diff_);
    clReleaseMemObject(sub_tmp_);
    clReleaseMemObject(sub_bottom_);

}

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
                                   
