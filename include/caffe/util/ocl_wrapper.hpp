// Copyright 2014 AMD DNN contributors.

#ifndef _CAFFE_UTIL_OCL_WRAPPER_HPP_
#define _CAFFE_UTIL_OCL_WRAPPER_HPP_

namespace caffe {

template <typename Dtype>
void get_max_gpu( const int num, const int dim, const Dtype* bottom_data, Dtype* scale_data);

template <typename Dtype>
void exp_gpu(const int num, const Dtype* data, Dtype* out);

template <typename Dtype>
void softmax_div_gpu(const int num, const int dim, const Dtype* scale, Dtype* data);

template <typename Dtype>
void scal_gpu(const int num, const Dtype alpha, Dtype* data);

template <typename Dtype>
void diff_gpu(const int num, const int dim, Dtype* data, const Dtype* label);
}  // namespace caffe

#endif  // CAFFE_UTIL_OCL_UTIL_HPP_
