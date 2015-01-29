// Copyright 2014 AMD DNN contributors.

#ifndef _CAFFE_UTIL_OCL_WRAPPER_HPP_
#define _CAFFE_UTIL_OCL_WRAPPER_HPP_

namespace caffe {

template <typename Dtype>
void get_max_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* bottom_data, Dtype* scale_data);

template <typename Dtype>
void exp_gpu(cl_kernel Kernel, const int num, const Dtype* data, Dtype* out);

template <typename Dtype>
void softmax_div_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* scale, Dtype* data);

template <typename Dtype>
void scal_gpu(cl_kernel Kernel, const int num, const Dtype alpha, Dtype* data);

template <typename Dtype>
void diff_gpu(cl_kernel Kernel, const int num, const int dim, Dtype* data, const Dtype* label);

template <typename Dtype>
void max_pool_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, Dtype* top_data);

template <typename Dtype>
void ave_pool_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, Dtype* top_data);

template <typename Dtype>
void max_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const Dtype* top_data, const Dtype* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, Dtype* bottom_diff );

template <typename Dtype>
void ave_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff, const int clnum, const int channels_, const int intheight_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, Dtype* bottom_diff);

template <typename Dtype>
void Relu_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, Dtype* top_data);

template <typename Dtype>
void Relu_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff);

template <typename Dtype>
void caffe_gpu_div (cl_kernel Kernel, const int n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void Dropout_fp_gpu(cl_kernel kernel, const int count, const Dtype* bottom_data, const int* MaskMem, const Dtype scale_, Dtype *top_data);

template <typename Dtype>
void Dropout_bp_gpu(cl_kernel kernel, const int count, const Dtype* top_diff, const int* MaskMem, const float threshold_, const Dtype scale_, Dtype* bottom_diff);

template <typename Dtype>
void caffe_gpu_bernoulli(cl_kernel ker_rand, int* a, const unsigned int n, Dtype inf, Dtype sup, Dtype threshold);
}  // namespace caffe

#endif  // CAFFE_UTIL_OCL_UTIL_HPP_
