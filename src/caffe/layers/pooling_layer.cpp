// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ocl_util.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void PoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "PoolingLayer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "PoolingLayer takes a single blob as output.";
  kernel_size_ = this->layer_param_.pooling_param().kernel_size();
  stride_ = this->layer_param_.pooling_param().stride();
  pad_ = this->layer_param_.pooling_param().pad();
  if (pad_ != 0) {
    CHECK_EQ(this->layer_param_.pooling_param().pool(),
             PoolingParameter_PoolMethod_AVE)
        << "Padding implemented only for average pooling.";
  }
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
  (*top)[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
Dtype PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  int count = (*top)[0]->count();
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:{

    //max_pool_fp();
    cl_int _err=0;    
    cl_kernel Kernel = clCreateKernel(amdDevice.Program,"MaxPoolForwardfloat",&_err);
    if(NULL == Kernel){
        fprintf(stderr,"Failed to create kernel %d\n",_err);
    }
    int clnum = bottom[0]->num();    

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&count);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&bottom_data);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&clnum);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&channels_);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&height_);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&width_);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&pooled_height_);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&pooled_width_);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&kernel_size_);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&stride_);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_mem),(void*)&top_data);
    OCL_CHECK(ret);

    size_t Global_Work_Size[]={count * 1};
    size_t Local_Work_Size[]={256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL, Global_Work_Size, Local_Work_Size,0,NULL,NULL));
    clReleaseKernel(Kernel);
#ifdef Track_layer
    LOG(WARNING) << "Max pool fp done";
#endif
    break;
  } 

  case PoolingParameter_PoolMethod_AVE:{
    //ave_pool_fp();
    cl_int _err = 0;
    cl_kernel Kernel = clCreateKernel(amdDevice.Program,"AvePoolForwardfloat",&_err);
    if(NULL == Kernel){
        fprintf(stderr,"Failed to create kernel %d\n", _err);
    }
    int clnum = bottom[0]->num();

    cl_int ret;
    ret = clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&count);
    ret |= clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&bottom_data);
    ret |= clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&clnum);
    ret |= clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&channels_);
    ret |= clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&height_);
    ret |= clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&width_);
    ret |= clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&stride_);
    ret |= clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&pad_);
    ret |= clSetKernelArg(Kernel,11,sizeof(cl_mem),(void*)&top_data);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[]={count * 1};
    size_t uiLocal_Work_Size[]={256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,NULL));
    clReleaseKernel(Kernel);

#ifdef Track_layer
    LOG(WARNING) << "Avg pool fp done";
#endif
    break;
  }
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  return Dtype(0.);
}

template <typename Dtype>
Dtype PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  int top_count = (*top)[0]->count();
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = -FLT_MAX;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = min(hstart + kernel_size_, height_);
            int wend = min(wstart + kernel_size_, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] =
                  max(top_data[ph * pooled_width_ + pw],
                      bottom_data[h * width_ + w]);
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_ - pad_;
            int wstart = pw * stride_ - pad_;
            int hend = min(hstart + kernel_size_, height_ + pad_);
            int wend = min(wstart + kernel_size_, width_ + pad_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += (*top)[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  return Dtype(0.);
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  int count = (*bottom)[0]->count();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:{
    //max_pool_bp();
    // The main loop
    cl_int _err=0;
    cl_kernel Kernel = clCreateKernel(amdDevice.Program,"MaxPoolBackwardfloat",&_err);
    if(NULL == Kernel){
        fprintf(stderr,"Failed to create kernel %d\n",_err);
    }
    int clnum = top[0]->num();

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&count);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&bottom_data);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_mem),(void*)&top_data);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_mem),(void*)&top_diff);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&clnum);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&channels_);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&height_);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&width_);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&pooled_height_);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&pooled_width_);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&kernel_size_);
    ret|=clSetKernelArg(Kernel,11,sizeof(cl_int),(void*)&stride_);
    ret|=clSetKernelArg(Kernel,12,sizeof(cl_mem),(void*)&bottom_diff);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[]={count};
    size_t uiLocal_Work_Size[]={64};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,NULL));
    clReleaseKernel(Kernel);

#ifdef Track_layer
    LOG(WARNING) << "Max pool bp done";
#endif
    break;
  }
  case PoolingParameter_PoolMethod_AVE:{
    
    //max_pool_bp();
    cl_int _err=0;
    cl_kernel Kernel = clCreateKernel(amdDevice.Program,"AvePoolBackwardfloat",&_err);
    if(NULL == Kernel){
        fprintf(stderr,"Failed to create kernel %d\n",_err);
    }
    int clnum = top[0]->num();

    cl_int ret;
    ret = clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&count);
    ret |= clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&top_diff);
    ret |= clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&clnum);
    ret |= clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&channels_);
    ret |= clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&height_);
    ret |= clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&width_);
    ret |= clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&stride_);
    ret |= clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&pad_);
    ret |= clSetKernelArg(Kernel,11,sizeof(cl_mem),(void*)&bottom_diff);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[]={count};
    size_t uiLocal_Work_Size[]={64};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,NULL));
    clReleaseKernel(Kernel);
#ifdef Track_layer
    LOG(WARNING) << "AVE pool bp done";
#endif

    break;
  }
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  memset(bottom_diff, 0, (*bottom)[0]->count() * sizeof(Dtype));
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_;
            int wstart = pw * stride_;
            int hend = min(hstart + kernel_size_, height_);
            int wend = min(wstart + kernel_size_, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                    top_diff[ph * pooled_width_ + pw] *
                    (bottom_data[h * width_ + w] ==
                        top_data[ph * pooled_width_ + pw]);
              }
            }
          }
        }
        // offset
        bottom_data += (*bottom)[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_ - pad_;
            int wstart = pw * stride_ - pad_;
            int hend = min(hstart + kernel_size_, height_ + pad_);
            int wend = min(wstart + kernel_size_, width_ + pad_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_data += (*bottom)[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_diff += (*bottom)[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}



INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
