// Copyright 2014 AMD DNN contributors.

#ifndef _CAFFE_UTIL_OCL_UTIL_HPP_
#define _CAFFE_UTIL_OCL_UTIL_HPP_

namespace caffe {

template <typename Dtype>
void ocl_memset(const Dtype* buffer, const Dtype value, const int count);


}  // namespace caffe

#endif  // CAFFE_UTIL_OCL_UTIL_HPP_
