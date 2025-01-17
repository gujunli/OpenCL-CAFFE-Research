// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ocl_util.hpp"
#include "caffe/util/ocl_wrapper.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype> 
PoolingLayer<Dtype>::~PoolingLayer(){
  OCL_CHECK( clReleaseKernel(MaxPoolForward_kernel) );
  OCL_CHECK( clReleaseKernel(AvePoolForward_kernel) );
  OCL_CHECK( clReleaseKernel(MaxPoolBackward_kernel) );
  OCL_CHECK( clReleaseKernel(AvePoolBackward_kernel) );
}

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
  //Intialize OpenCL related
  ocl_setup();
}

template <typename Dtype>
 void PoolingLayer<Dtype>::ocl_setup(){
  MaxPoolForward_kernel = clCreateKernel(amdDevice.Program, "MaxPoolForwardfloat", NULL);
  AvePoolForward_kernel = clCreateKernel(amdDevice.Program, "AvePoolForwardfloat", NULL);
  MaxPoolBackward_kernel = clCreateKernel(amdDevice.Program, "MaxPoolBackwardfloat", NULL);
  AvePoolBackward_kernel = clCreateKernel(amdDevice.Program, "AvePoolBackwardfloat", NULL);
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
  int clnum = bottom[0]->num();    
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:{
    max_pool_fp_gpu(MaxPoolForward_kernel, count, bottom_data, clnum, channels_, height_, width_, pooled_height_, pooled_width_, kernel_size_, stride_, top_data);
#ifdef Track_layer
    LOG(WARNING) << "Max pool fp done";
#endif
    break;
  } 
  case PoolingParameter_PoolMethod_AVE:{
    ave_pool_fp_gpu(AvePoolForward_kernel, count, bottom_data, clnum, channels_, height_,width_, pooled_height_, pooled_width_, kernel_size_, stride_, pad_, top_data);
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
  int clnum = top[0]->num();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:{
    max_pool_bp_gpu(MaxPoolBackward_kernel, count, bottom_data, top_data, top_diff, clnum, channels_, height_, width_, pooled_height_, pooled_width_, kernel_size_, stride_, bottom_diff );
#ifdef Track_layer
    LOG(WARNING) << "Max pool bp done";
#endif
    break;
  }
  case PoolingParameter_PoolMethod_AVE:{
    ave_pool_bp_gpu(AvePoolBackward_kernel, count, top_diff, clnum, channels_, height_, width_, pooled_height_, pooled_width_, kernel_size_, stride_, pad_, bottom_diff);
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
Dtype PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  int top_count = (*top)[0]->count();
#ifdef Track_layer
    LOG(WARNING) << "pool fp done";
#endif
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
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }
#ifdef Track_layer
    LOG(WARNING) << "pool bp done";
#endif
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
