// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ocl_util.hpp"
#include "caffe/util/ocl_wrapper.hpp"

using std::max;

namespace caffe {
template <typename Dtype>
void PowerLayer<Dtype>::ocl_setup(){
   memset_kernel = clCreateKernel(amdDevice.Program, "oclmemfloat", NULL);
   scalar_kernel = clCreateKernel(amdDevice.Program, "add_scalar_float", NULL);
   div_kernel = clCreateKernel(amdDevice.Program, "div_float", NULL);
   powx_kernel = clCreateKernel(amdDevice.Program, "powx_float", NULL);
   mul_kernel = clCreateKernel(amdDevice.Program, "element_mul_float", NULL);
}

template <typename Dtype>
PowerLayer<Dtype>::~PowerLayer(){
    OCL_CHECK( clReleaseKernel(memset_kernel) );
    OCL_CHECK( clReleaseKernel(scalar_kernel) );
    OCL_CHECK( clReleaseKernel(div_kernel) );
    OCL_CHECK( clReleaseKernel(powx_kernel) );
    OCL_CHECK( clReleaseKernel(mul_kernel) );

}

template <typename Dtype>
void PowerLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  power_ = this->layer_param_.power_param().power();
  scale_ = this->layer_param_.power_param().scale();
  shift_ = this->layer_param_.power_param().shift();
  diff_scale_ = power_  * scale_;
  //OpenCL related set up
  ocl_setup();
}

// Compute y = (shift + scale * x)^power
template <typename Dtype>
Dtype PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//Dtype PowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  //LOG(INFO) << "Power layer fp cpu";
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    caffe_set(count, value, top_data);
    return Dtype(0);
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    caffe_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_add_scalar(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_powx(count, top_data, power_, top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//void PowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  LOG(INFO) << "Power layer bp cpu";
  if (propagate_down) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      caffe_set(count, diff_scale_, bottom_diff);
    } else {
      const Dtype* bottom_data = (*bottom)[0]->cpu_data();
      // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
      //               = diff_scale * y / (shift + scale * x)
      if (power_ == Dtype(2)) {
        // Special case for y = (shift + scale * x)^2
        //     -> dy/dx = 2 * scale * (shift + scale * x)
        //              = diff_scale * shift + diff_scale * scale * x
        caffe_cpu_axpby(count, diff_scale_ * scale_, bottom_data,
            Dtype(0), bottom_diff);
        if (shift_ != Dtype(0)) {
          caffe_add_scalar(count, diff_scale_ * shift_, bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        // Special case for y = (scale * x)^power
        //     -> dy/dx = scale * power * (scale * x)^(power - 1)
        //              = scale * power * (scale * x)^power * (scale * x)^(-1)
        //              = power * y / x
        const Dtype* top_data = top[0]->cpu_data();
        caffe_div(count, top_data, bottom_data, bottom_diff);
        caffe_scal(count, power_, bottom_diff);
      } else {
        caffe_copy(count, bottom_data, bottom_diff);
        if (scale_ != Dtype(1)) {
          caffe_scal(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          caffe_add_scalar(count, shift_, bottom_diff);
        }
        const Dtype* top_data = top[0]->cpu_data();
        caffe_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          caffe_scal(count, diff_scale_, bottom_diff);
        }
      }
    }
    if (diff_scale_ != Dtype(0)) {
      caffe_mul(count, top_diff, bottom_diff, bottom_diff);
    }
  }
}

template <typename Dtype>
Dtype PowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//Dtype PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "Power Layer fp_GPU";
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    //caffe_gpu_set(count, value, top_data);
    ocl_memset(memset_kernel, top_data, value, count);
    return Dtype(0);
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_gpu_copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    caffe_gpu_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_gpu_add_scalar(scalar_kernel, count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_gpu_powx(powx_kernel, count, top_data, power_, top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//void PowerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  LOG(INFO) << "Power Layer bp_GPU";
  if (propagate_down) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      caffe_gpu_set(count, diff_scale_, bottom_diff);
    } else {
      const Dtype* bottom_data = (*bottom)[0]->gpu_data();
      // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
      //               = diff_scale * y / (shift + scale * x)
      if (power_ == Dtype(2)) {
        // Special case for y = (shift + scale * x)^2
        //     -> dy/dx = 2 * scale * (shift + scale * x)
        //              = diff_scale * shift + diff_scale * scale * x
        caffe_gpu_axpby(count, diff_scale_ * scale_, bottom_data,
            Dtype(0), bottom_diff);
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(scalar_kernel, count, diff_scale_ * shift_, bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        // Special case for y = (scale * x)^power
        //     -> dy/dx = scale * power * (scale * x)^(power - 1)
        //              = scale * power * (scale * x)^power * (scale * x)^(-1)
        //              = power * y / x
        const Dtype* top_data = top[0]->gpu_data();
        caffe_gpu_div(div_kernel, count, top_data, bottom_data, bottom_diff);
        caffe_gpu_scal(count, power_, bottom_diff);
      } else {
        caffe_gpu_copy(count, bottom_data, bottom_diff);
        if (scale_ != Dtype(1)) {
          caffe_gpu_scal(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(scalar_kernel, count, shift_, bottom_diff);
        }
        const Dtype* top_data = top[0]->gpu_data();
        caffe_gpu_div(div_kernel, count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          caffe_gpu_scal(count, diff_scale_, bottom_diff);
        }
      }
    }
    caffe_gpu_mul(mul_kernel, count, top_diff, bottom_diff, bottom_diff);
  }
}


INSTANTIATE_CLASS(PowerLayer);


}  // namespace caffe
