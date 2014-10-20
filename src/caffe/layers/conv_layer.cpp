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
void ConvolutionLayer<Dtype>::ocl_setup(const int bottom0_offset1,
     const int top0_offset1) {
 //create OpenCL related cl_mem objects and kernels
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  sub_top = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(top_offset*sizeof(Dtype)), NULL, NULL);
  sub_weight = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(weight_offset*sizeof(Dtype)), NULL, NULL);
  sub_bottom = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(bottom0_offset1)*sizeof(Dtype), NULL, NULL);
  sub_im2col = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(col_offset)*sizeof(Dtype), NULL, NULL);
  sub_top_diff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(top0_offset1*sizeof(Dtype)), NULL, NULL);
  sub_col2im = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)((bottom0_offset1)*sizeof(Dtype)), NULL, NULL);

  im2col_kernel = clCreateKernel(amdDevice.Program,"im2colfloat", NULL);
  col2im_kernel = clCreateKernel(amdDevice.Program,"col2imfloat", NULL);
  //CHECK_EQ(col2im_kernel, NULL) << "failed to create col2im_kernel";

}


template <typename Dtype>
  ConvolutionLayer<Dtype>::~ConvolutionLayer(){
  OCL_CHECK( clReleaseMemObject(sub_top) );
  OCL_CHECK( clReleaseMemObject(sub_weight) );
  OCL_CHECK( clReleaseMemObject(sub_bottom) );
  OCL_CHECK( clReleaseMemObject(sub_im2col) );
  OCL_CHECK( clReleaseMemObject(sub_top_diff) );
  OCL_CHECK( clReleaseMemObject(sub_col2im) );
  OCL_CHECK( clReleaseMemObject(sub_bottom) );
  OCL_CHECK( clReleaseKernel(im2col_kernel) );
  OCL_CHECK( clReleaseKernel(col2im_kernel) );
}


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

    //initializa OpenCL kernels and cl_mem objects
    ocl_setup(bottom[0]->offset(1), (*top)[0]->offset(1));

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
Dtype ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  for (int n = 0; n < num_; ++n) {
    // First, im2col
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)bottom_data, (cl_mem)sub_bottom, (size_t)(bottom[0]->offset(n)*sizeof(Dtype)), 0, bottom[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
    
    im2col_gpu(im2col_kernel, (cl_mem)sub_bottom, channels_, height_, 
                       width_, kernel_size_, pad_, stride_, col_data);
    //note if(group!=1) it doesn't work
    for (int g = 0; g < group_; ++g) {
      OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)col_data, (cl_mem)sub_im2col, (size_t)(col_offset * g * sizeof(Dtype)), 0, col_offset *sizeof(Dtype), 0, NULL, NULL));

      OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)weight, (cl_mem)sub_weight, (size_t)(weight_offset * g * sizeof(Dtype)), 0, weight_offset * sizeof(Dtype), 0, NULL, NULL));

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., (Dtype*)sub_weight, (Dtype*)sub_im2col,
        (Dtype)0., (Dtype*)sub_top);
    }

    // third, add bias
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          (Dtype)1., (Dtype*)sub_top);
    }
    
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)sub_top, (cl_mem)top_data, 0, (size_t)((*top)[0]->offset(n) * sizeof(Dtype)), ((*top)[0]->count()/num_)*sizeof(Dtype), 0, NULL, NULL));
  }
#ifdef Track_layer
  LOG(WARNING) << "conv fp done";
#endif
  return Dtype(0.);
}
    
template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           vector<Blob<Dtype>*>* top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = (*top)[0]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < num_; ++n) {
        // First, im2col
        im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
                   width_, kernel_size_, pad_, stride_, (Dtype*) col_data);
        // Second, innerproduct with groups
        for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
                                  (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
                                  (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
        }
        // third, add bias
        if (bias_term_) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                                  N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
                                  reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
                                  (Dtype)1., top_data + (*top)[0]->offset(n));
        }
    }
    return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  Dtype* bias_diff = NULL;

  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    ocl_memset(bias_diff, (Dtype)(0.), this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemvv<Dtype>(CblasNoTrans, M_, N_,
          (Dtype)1., top_diff, top[0]->offset(n), N_,
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), (size_t)0, (Dtype)1., 1,
          bias_diff, (size_t)0, 1);
    }
  }
  
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  cl_int err;
  
  for (int n = 0; n < num_; ++n) {
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)top_diff, sub_top_diff, (size_t)(top[0]->offset(n)*sizeof(Dtype)), 0, (top[0]->offset(1)*sizeof(Dtype)), 0, NULL, NULL));
    OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)bottom_data, sub_bottom, (size_t)((*bottom)[0]->offset(n)*sizeof(Dtype)), 0, (*bottom)[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_gpu(im2col_kernel, sub_bottom, channels_, height_,
                      width_, kernel_size_, pad_, stride_, (Dtype*) col_data);
   
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., (Dtype*)sub_top_diff,
        (Dtype*)col_data, (Dtype)1.,
        (Dtype*)weight_diff);
    }

   if (propagate_down) {
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight,
          (Dtype*)sub_top_diff,
          (Dtype)0., col_diff);
      }
    
      // col2im back to the data
      col2im_gpu(col2im_kernel, (cl_mem)col_diff, channels_, height_, width_, kernel_size_, pad_,
          stride_, (Dtype*)sub_col2im);
      OCL_CHECK(clEnqueueCopyBuffer(amdDevice.CommandQueue, sub_col2im, (cl_mem)bottom_diff, 0, (size_t)((*bottom)[0]->offset(n)*sizeof(Dtype)), (*bottom)[0]->offset(1)*sizeof(Dtype), 0, NULL, NULL));
    }

  }
#ifdef Track_layer
    LOG(WARNING) << "conv bp done";
#endif

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
    for (int n = 0; n < num_; ++n) {
        // since we saved memory in the forward pass by not storing all col data,
        // we will need to recompute them.
        im2col_cpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
                   width_, kernel_size_, pad_, stride_, col_data);
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
}

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
                                   
