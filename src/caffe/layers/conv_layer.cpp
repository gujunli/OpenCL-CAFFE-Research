// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

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
  cl_mem subColMem=clCreateBuffer(amdDevice.Context,CL_MEM_READ_WRITE, (size_t)(col_offset)*sizeof(Dtype), NULL, NULL);
  for (int n = 0; n < num_; ++n) {
    // First, im2col
    cl_int iStatus;
    iStatus = clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)bottom_data, (cl_mem)subBotMem, (size_t)(bottom[0]->offset(n)*sizeof(Dtype)), 0, bottom[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL);
    
    im2col_gpu(subBotMem, channels_, height_, 
                       width_, kernel_size_, pad_, stride_, col_data);
    // Second, innerproduct with groups
    //int height_col = (height_ +2*pad_-kernel_size_)/stride_+1;
    //int width_col = (width_ +2*pad_-kernel_size_)/stride_+1;

    //note if(group!=1) it doesn't work
        
    for (int g = 0; g < group_; ++g) {
      iStatus = clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)col_data, (cl_mem)subColMem, (size_t)(col_offset * g * sizeof(Dtype)), 0, col_offset *sizeof(Dtype), 0, NULL, NULL);

      iStatus = clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)weight, (cl_mem)subWgtMem, (size_t)(weight_offset * g * sizeof(Dtype)), 0, weight_offset * sizeof(Dtype), 0, NULL, NULL);

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
    
    iStatus = clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)subTopMem, (cl_mem)top_data, 0, (size_t)((*top)[0]->offset(n) * sizeof(Dtype)), ((*top)[0]->count()/num_)*sizeof(Dtype), 0, NULL, NULL);
  }
  clReleaseMemObject(subTopMem);
  clReleaseMemObject(subWgtMem);
  clReleaseMemObject(subBotMem);
  clReleaseMemObject(subColMem);
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_data_ = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
          bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  cl_mem tmpmem=clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)((*bottom)[0]->offset(1))*sizeof(Dtype), NULL, NULL);
  for (int n = 0; n < num_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    cl_int iStatus=clEnqueueWriteBuffer(amdDevice.CommandQueue, tmpmem, CL_TRUE,0,(*bottom)[0]->offset(1)*sizeof(Dtype),(const void*)(bottom_data+(*bottom)[0]->offset(n)),0,NULL,NULL);
    if(iStatus!=CL_SUCCESS){
        fprintf(stderr,"Failed to EnqueueWriteBuffer\n");
    }
    clFlush(amdDevice.CommandQueue);
    
    im2col_gpu(tmpmem, channels_, height_, width_, kernel_size_, pad_, stride_, col_data_);
    
    int height_col = (height_ +2*pad_-kernel_size_)/stride_+1;
    int width_col = (width_ +2*pad_-kernel_size_)/stride_+1;
    iStatus = clEnqueueReadBuffer(amdDevice.CommandQueue, (cl_mem)col_data_, CL_TRUE, 0, kernel_size_*kernel_size_*channels_*height_col*width_col*sizeof(Dtype), col_data, 0, NULL, NULL);
    //im2col_cpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
    //                  width_, kernel_size_, pad_, stride_, col_data);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        weight_diff + weight_offset * g);
    }
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down) {
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff + col_offset * g);
      }
      // col2im back to the data
      col2im_cpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
          stride_, bottom_diff + (*bottom)[0]->offset(n));
    }
  }
  clReleaseMemObject(tmpmem);
}

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
