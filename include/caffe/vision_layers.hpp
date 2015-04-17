// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
ConcatLayer
  Takes at least two blobs and concatenates them along either num or
  channel dim, outputting the result.
*/
template <typename Dtype>
class ConcatLayer : public Layer<Dtype> {
 public:
  explicit ConcatLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> col_bob_;
  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int concat_dim_;
};

/* ConvolutionLayer
*/
template <typename Dtype>
class ConvolutionLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual  ~ConvolutionLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  void ocl_setup(const int bottom_offset, const int top_offset); 
      //const int bottom0_offset_optnum, const int top0_offset_optnum);
  int kernel_size_;
  int stride_;
  int num_;
  int channels_;
  int pad_;
  int height_;
  int width_;
  int num_output_;
  int group_;
  Blob<Dtype> col_buffer_;
  shared_ptr<SyncedMemory> bias_multiplier_;
  bool bias_term_;
  int M_;
  int K_;
  int N_;

//opencl related data structures
protected:
  cl_kernel im2col_kernel, col2im_kernel;
  cl_kernel oclmem_kernel;
  cl_kernel ocl_Kernel_im2colfloat, ocl_Kernel_col2imfloat;
  cl_kernel ocl_Kernel_transpose, ocl_Kernel_transform;
  cl_kernel im2col_16_kernel;
  cl_mem subTopMem, transMem;
};

/* EltwiseProductLayer
*/
template <typename Dtype>
class EltwiseProductLayer : public Layer<Dtype> {
 public:
  explicit EltwiseProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~EltwiseProductLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  void ocl_setup(const int num);

//opencl related data structures
protected:
  //cl_mem scratch_buf;
  cl_kernel div_kernel; 
  cl_kernel mul_kernel; 
};

template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
 public:
  explicit FlattenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  int count_;
};

/* Im2colLayer
*/
template <typename Dtype>
class Im2colLayer : public Layer<Dtype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  ~Im2colLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  void ocl_setup();

  int kernel_size_;
  int stride_;
  int channels_;
  int height_;
  int width_;
  int pad_;

  cl_mem col_data, sub_bottom, sub_top_diff, sub_bottom_diff;
  cl_kernel im2col_kernel, col2im_kernel;
};

/* InnerProductLayer
*/
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  shared_ptr<SyncedMemory> bias_multiplier_;
};

// Forward declare PoolingLayer and SplitLayer for use in LRNLayer.
template <typename Dtype> class PoolingLayer;
template <typename Dtype> class SplitLayer;

/* LRNLayer
  Local Response Normalization
*/
template <typename Dtype>
class LRNLayer : public Layer<Dtype> {
 public:
  explicit LRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  virtual Dtype CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype WithinChannelForward(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void WithinChannelBackward(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  int num_;
  int channels_;
  int height_;
  int width_;

  cl_kernel LFSkernel;
  cl_kernel LCDkernel;
  cl_kernel LCOkernel;

  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
  Blob<Dtype> scale_;

  // Fields used for normalization WITHIN_CHANNEL
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  vector<Blob<Dtype>*> split_top_vec_;
  shared_ptr<PowerLayer<Dtype> > square_layer_;
  Blob<Dtype> square_input_;
  Blob<Dtype> square_output_;
  vector<Blob<Dtype>*> square_bottom_vec_;
  vector<Blob<Dtype>*> square_top_vec_;
  shared_ptr<PoolingLayer<Dtype> > pool_layer_;
  Blob<Dtype> pool_output_;
  vector<Blob<Dtype>*> pool_top_vec_;
  shared_ptr<PowerLayer<Dtype> > power_layer_;
  Blob<Dtype> power_output_;
  vector<Blob<Dtype>*> power_top_vec_;
  shared_ptr<EltwiseProductLayer<Dtype> > product_layer_;
  Blob<Dtype> product_data_input_;
  vector<Blob<Dtype>*> product_bottom_vec_;
};

/* PoolingLayer
*/
template <typename Dtype>
class MemoryDataLayer : public Layer<Dtype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, Dtype* label, int n);
  int datum_channels() { return datum_channels_; }
  int datum_height() { return datum_height_; }
  int datum_width() { return datum_width_; }
  int batch_size() { return batch_size_; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) { return; }

  Dtype* data_;
  Dtype* labels_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  int batch_size_;
  int n_;
  int pos_;
};

template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  ~PoolingLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  void ocl_setup();

  int kernel_size_;
  int stride_;
  int pad_;
  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Blob<Dtype> rand_idx_;
//opencl related data structures
protected:
  cl_kernel MaxPoolForward_kernel, 
	    AvePoolForward_kernel, 
            MaxPoolBackward_kernel, 
            AvePoolBackward_kernel;
 
};

/* SoftmaxLayer
*/
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  ~SoftmaxLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  void ocl_setup();

  // sum_multiplier is just used to carry out sum using blas
  Blob<Dtype> sum_multiplier_;
  // scale is an intermediate blob to hold temporary results.
  Blob<Dtype> scale_;
 protected:
  cl_kernel get_max_kernel, exp_kernel, softmax_div_kernel;
};

/* SoftmaxWithLossLayer
  Implements softmax and computes the loss.

  It is preferred over separate softmax + multinomiallogisticloss
  layers due to more numerically stable gradients.

  In test, this layer could be replaced by simple softmax layer.
*/
template <typename Dtype>
class SoftmaxWithLossLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxWithLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
   ~SoftmaxWithLossLayer();
 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  void ocl_setup();

  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  // prob stores the output probability of the layer.
  Blob<Dtype> prob_;
  // Vector holders to call the underlying softmax layer forward and backward.
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
 protected:
   cl_kernel diff_kernel, scal_kernel, softmax_kernel;
   cl_mem d_loss; 
};

/* SplitLayer
*/
template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  int count_;
};

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
