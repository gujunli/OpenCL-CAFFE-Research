// Copyright 2014 BVLC and contributors.

#include <vector>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ocl_util.hpp"
#include "caffe/util/ocl_wrapper.hpp"
#include <CL/cl.h>

namespace caffe {
template <typename Dtype>
void ConvolutionLayer<Dtype>::ocl_setup(const int bottom0_offset1,
     const int top0_offset1) {
  im2col_kernel = clCreateKernel(amdDevice.Program,"im2colfloat", NULL);
  col2im_kernel = clCreateKernel(amdDevice.Program,"col2imfloat", NULL);
  oclmem_kernel = clCreateKernel(amdDevice.Program, "oclmemfloat", NULL);
  //CHECK_EQ(col2im_kernel, NULL) << "failed to create col2im_kernel";
  im2col_16_kernel = clCreateKernel(amdDevice.Program, "im2col_16float", NULL);
  ocl_Kernel_im2colfloat = clCreateKernel(amdDevice.Program,"im2colfloat_yuan",NULL);
  ocl_Kernel_col2imfloat = clCreateKernel(amdDevice.Program,"col2imfloat_yuan",NULL);
  ocl_Kernel_transpose = clCreateKernel(amdDevice.Program,"transposefloat",NULL);
  ocl_Kernel_transform = clCreateKernel(amdDevice.Program,"transformfloat",NULL);
  tmpMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)( (K_ * group_) * N_ * opt_num * sizeof(Dtype)), NULL, NULL);
  subTopMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)((M_ * group_) * N_ * opt_num * sizeof(Dtype)), NULL, NULL);
  subTopMem2 = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)((M_ * group_) * N_ * opt_num * sizeof(Dtype)), NULL, NULL);
  transMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)((K_ *group_ )* N_ * opt_num * sizeof(Dtype)), NULL, NULL);
  subBotMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(bottom0_offset1*opt_num)*sizeof(Dtype), NULL, NULL);
  subBotdiff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(bottom0_offset1*opt_num)*sizeof(Dtype), NULL, NULL);
  subTopdiff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, (size_t)(top0_offset1*opt_num)*sizeof(Dtype), NULL, NULL);

}


template <typename Dtype>
 ConvolutionLayer<Dtype>::~ConvolutionLayer(){
 //if(Caffe::mode() == Caffe::GPU){
  OCL_CHECK( clReleaseKernel(im2col_kernel) );
  OCL_CHECK( clReleaseKernel(col2im_kernel) );
  OCL_CHECK( clReleaseKernel(oclmem_kernel) );
  OCL_CHECK( clReleaseKernel(ocl_Kernel_transpose) );
  OCL_CHECK( clReleaseKernel(ocl_Kernel_transform) );
  OCL_CHECK( clReleaseKernel(ocl_Kernel_im2colfloat) );
  OCL_CHECK( clReleaseKernel(ocl_Kernel_col2imfloat) );
  OCL_CHECK( clReleaseKernel(im2col_16_kernel) );
  //}
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
   //if(Caffe::mode() == Caffe::GPU)
     //{
       ocl_setup(bottom[0]->offset(1), (*top)[0]->offset(1));
      //LOG(INFO) << "conv ocl_setup: uses GPU already set up";
     //}

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
  cl_event profEvent;
#if defined (use_yuan_scheme)
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  //int col_size = (K_ * group_ ) * N_;
  int top_offset = M_ * N_;
  int opt_num2 = opt_num;
  //printf("M_=%d, N_=%d, K_=%d, weight_offset=%d col_offset = %d, top_offset =%d \n", M_, N_, K_, weight_offset, col_offset, top_offset);
  for (int n = 0; n < num_; n += opt_num2) {
    opt_num2 = opt_num2 > (num_ - n)? (num_ - n) : opt_num2;
   //for the last loop, may not be 16. for correctness, col_offset, weight_offset, top_offset will all be different
    top_offset = M_ * N_ * opt_num2;
    col_offset = K_ * N_ * opt_num2;  
    // First, im2col
    //this should be opt_num2 images packing together.
    im2col_16_gpu(im2col_16_kernel, bottom_data, bottom[0]->offset(n), channels_, height_, 
                       width_, kernel_size_, pad_, stride_, (Dtype*)transMem, 0);
   
 #ifdef pipeline
    int g = 0;
    if(group_ == 1){
       profEvent = caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_ * opt_num2, K_,
          (Dtype)1., weight, weight_offset * g, (Dtype*)transMem, col_offset * g,
          (Dtype)0., (Dtype*)subTopMem, top_offset * g); 
    }else if (group_ == 2){
       g = 0;
       caffe_gpu_gemmex<Dtype>(&(amdDevice.CommandQueue), CblasNoTrans, CblasNoTrans, M_, N_ * opt_num2, K_,
          (Dtype)1., weight, weight_offset * g, (Dtype*)transMem, col_offset * g,
          (Dtype)0., (Dtype*)subTopMem, top_offset * g); 
 
       g = 1;
       caffe_gpu_gemmex<Dtype>(&(amdDevice.CommandQueue_helper), CblasNoTrans, CblasNoTrans, M_, N_ * opt_num2, K_,
          (Dtype)1., weight, weight_offset * g, (Dtype*)transMem, col_offset * g,
          (Dtype)0., (Dtype*)subTopMem, top_offset * g); 
     }
#else
    for (int g = 0; g < group_; ++g) {
       profEvent = caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_ * opt_num2, K_,
          (Dtype)1., weight, weight_offset * g, (Dtype*)transMem, col_offset * g,
          (Dtype)0., (Dtype*)subTopMem, top_offset * g); 
       }
#endif
    transform_gpu(ocl_Kernel_transform, (Dtype*)subTopMem, top_data, (*top)[0]->offset(n), N_, M_, opt_num2);
    for (int z = 0; z < opt_num2; z++)
      if (bias_term_) {
        caffe_gpu_exgemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
            (Dtype)1., this->blobs_[1]->gpu_data(), reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
            (Dtype)1., (Dtype*)top_data, 0, 0, (*top)[0]->offset(n) + M_ * N_ * z);
    }
  }

#elif defined (use_packing_scheme)
  printf("using packing scheme \n");
  /*int the packing schme, M, K stay the same. N multiplies by opt_num becomes much bigger N'. 
   N' is the M in sgemm call.*/ 
  int weight_offset = M_ * K_;
  /*we still record the original M, N, K for data storage*/
  int M_org = M_ * group_;
  int K_org = K_ * group_;
  int N_org = N_;
  //col_offset is the offset for sgemm, including packing and groups
  int col_offset = K_ * (N_ * opt_num);
  int top_offset = M_ * (N_ * opt_num);
  // col_size the size of one image's col_data
  int col_size = channels_ * kernel_size_ * kernel_size_ * N_; // N = height_out * width_out
  int opt_num2 = opt_num;
  for (int n = 0; n < num_; n += opt_num2) {
    opt_num2 = opt_num2 > (num_ - n)? (num_ - n) : opt_num2;
    // First, im2col
    for (int z = 0; z < opt_num2; z++){
      im2col_gpu(im2col_kernel, bottom_data, bottom[0]->offset(n+z), channels_, height_, 
                       width_, kernel_size_, pad_, stride_, col_data, 0);
      printf("M_=%d, N_=%d, K_=%d, col_size=%d \n", M_, N_, K_, col_size);
      printf("M_org=%d, N_org=%d, K_org=%d, col_size=%d \n", M_org, N_org, K_org, col_size);
      OCL_CHECK( clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)col_data, (cl_mem)tmpMem, 0, ( (col_size) * z * sizeof(Dtype)), col_size * sizeof(Dtype), 0, NULL, NULL) );
    }
  
      int ID = 0;
   //step 2: transpose
    cl_int ret;
    ret =clSetKernelArg(ocl_Kernel_transpose,0,sizeof(cl_mem),(void*)&tmpMem);
    ret|=clSetKernelArg(ocl_Kernel_transpose,1,sizeof(cl_mem),(void*)&transMem);
    ret|=clSetKernelArg(ocl_Kernel_transpose,2,sizeof(cl_int),(void*)&N_);
    ret|=clSetKernelArg(ocl_Kernel_transpose,3,sizeof(cl_int),(void*)&K_org);
    ret|=clSetKernelArg(ocl_Kernel_transpose,4,sizeof(cl_int),(void*)&opt_num2);
    cl_event eventPoint;
    size_t uiGlobal_Work_Size[2];
    uiGlobal_Work_Size[0]= N_;
    uiGlobal_Work_Size[1]= ((((K_*group_) *opt_num2)+255)/256)*256;
    size_t uiLocal_Work_Size[2];
    uiLocal_Work_Size[0]=256;
    uiLocal_Work_Size[1]=1;
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, ocl_Kernel_transpose, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, &eventPoint) );
 
   //step 3: sgemm: Top (subTopMem) = weight * col_data

    for (int g = 0; g < group_; ++g) {
       profEvent = caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_ * opt_num2, K_,
          (Dtype)1., weight, weight_offset * g, (Dtype*)transMem, col_offset * g,
          (Dtype)0., (Dtype*)subTopMem, top_offset * g); 
       }
   //step 4: tranform
    /*design: transform results can be put directly into top_data. will remove the last CopyBuffer overhead*/
    ret =clSetKernelArg(ocl_Kernel_transform,0,sizeof(cl_mem),(void*)&subTopMem);
    ret|=clSetKernelArg(ocl_Kernel_transform,1,sizeof(cl_mem),(void*)&subTopMem2);
    ret|=clSetKernelArg(ocl_Kernel_transform,2,sizeof(cl_int),(void*)&N_);
    ret|=clSetKernelArg(ocl_Kernel_transform,3,sizeof(cl_int),(void*)&M_org);
    ret|=clSetKernelArg(ocl_Kernel_transform,4,sizeof(cl_int),(void*)&opt_num2);
    size_t uiGlobal_Work_Size2[]={M_org * opt_num2};
    size_t uiLocal_Work_Size2[]={256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, ocl_Kernel_transform, 1, NULL,uiGlobal_Work_Size2, uiLocal_Work_Size2, 0, NULL, &eventPoint) );
   
    //step 5: add bias
    /*note: this sgemm has to use num_output_ instead of M, because M = M /group, in setup*/
    for (int z = 0; z < opt_num2; z++)
    if (bias_term_) {
      //profEvent = caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      profEvent = caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasNoTrans, M_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(), 0,
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), 0,
          (Dtype)1., (Dtype*) subTopMem2, M_ * N_*z);
      OCL_CHECK( clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem)subTopMem2, (cl_mem)top_data, 0, (M_ * N_ * n * sizeof(Dtype)), M_ * N_ * opt_num2 * sizeof(Dtype), 0, NULL, NULL) );
     }

  }

  /*scheme 2: original im2col + sgemm scheme with multi-commandQ*/  
#else
  printf("using original scheme \n");
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  for (int n = 0; n < num_; ++n) {
    // First, im2col
    im2col_gpu(im2col_kernel, bottom_data, bottom[0]->offset(n), channels_, height_, 
                       width_, kernel_size_, pad_, stride_, col_data, 0);
   
   // for (int g = 0; g < group_; ++g) {
 #ifdef pipeline
    int g = 0;
    if(group_ == 1){
      profEvent = caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
	(Dtype)1., weight, weight_offset * g, col_data, col_offset * g,
        (Dtype)0., top_data, (*top)[0]->offset(n) + top_offset * g);
    //}
    }else if (group_ == 2){
     //printf("cmdq: q1 %d, d2 %d\n", amdDevice.CommandQueue, amdDevice.CommandQueue_helper);
    g = 0;
      caffe_gpu_gemmex<Dtype>(&(amdDevice.CommandQueue), CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight, weight_offset * g, col_data, col_offset * g,
        (Dtype)0., top_data, (*top)[0]->offset(n) + top_offset * g);
 
    //printf("group =0 done \n");
    g = 1;
      caffe_gpu_gemmex<Dtype>(&(amdDevice.CommandQueue_helper), CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight, weight_offset * g, col_data, col_offset * g,
        (Dtype)0., top_data, (*top)[0]->offset(n) + top_offset * g);
    //printf("end: group =1 \n");
     }
 #else
 for (int g = 0; g < group_; ++g) {
   profEvent = caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight, weight_offset * g, col_data, col_offset * g,
        (Dtype)0., top_data, (*top)[0]->offset(n) + top_offset * g);
  }
 #endif
    // third, add bias
    if (bias_term_) {
      profEvent = caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(), 0,
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), 0,
          (Dtype)1., top_data, (*top)[0]->offset(n));
      int ID = 0;
      clSetEventCallback(profEvent, CL_COMPLETE, &eventCallback, (void*)ID);
    }

  }

#endif

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
    ocl_memset(oclmem_kernel, bias_diff, (Dtype)(0.), this->blobs_[1]->count());
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

  for (int n = 0; n < num_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_gpu(im2col_kernel, bottom_data, (*bottom)[0]->offset(n), channels_, height_,
                      width_, kernel_size_, pad_, stride_, col_data, 0);

#ifdef pipeline
    // gradient w.r.t. weight. Note that we will accumulate diffs.
     int g = 0;
     if(group_ == 1){
      caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff, top[0]->offset(n),
        (Dtype*)col_data, col_offset * g, (Dtype)1.,
        (Dtype*)weight_diff, weight_offset * g);
     }else if (group_ == 2){
      g = 0;
      caffe_gpu_gemmex<Dtype>(&(amdDevice.CommandQueue), CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff, top[0]->offset(n),
        (Dtype*)col_data, col_offset * g, (Dtype)1.,
        (Dtype*)weight_diff, weight_offset * g);
      g = 1;
      caffe_gpu_gemmex<Dtype>(&(amdDevice.CommandQueue_helper), CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff, top[0]->offset(n),
        (Dtype*)col_data, col_offset * g, (Dtype)1.,
        (Dtype*)weight_diff, weight_offset * g);
    }

   if (propagate_down) {
     int g = 0;
     if(group_ == 1){
        caffe_gpu_gemm_ex<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight,  weight_offset * g,
          top_diff, top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff, col_offset * g);
     }else if (group_ == 2){
        g = 0;
        caffe_gpu_gemmex<Dtype>(&(amdDevice.CommandQueue), CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight,  weight_offset * g,
          top_diff, top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff, col_offset * g);
        g = 1;
        caffe_gpu_gemmex<Dtype>(&(amdDevice.CommandQueue_helper), CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight,  weight_offset * g,
          top_diff, top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff, col_offset * g);
      }
    }
#else
for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm_ex<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff, top[0]->offset(n),
        (Dtype*)col_data, col_offset * g, (Dtype)1.,
        (Dtype*)weight_diff, weight_offset * g);
    }

   if (propagate_down) {
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm_ex<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight,  weight_offset * g,
          top_diff, top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff, col_offset * g);
      }
    }
#endif
    // col2im back to the data
      col2im_gpu(col2im_kernel, col_diff, 0, channels_, height_, width_, kernel_size_, pad_,
          stride_, bottom_diff, (*bottom)[0]->offset(n));
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
                                   
