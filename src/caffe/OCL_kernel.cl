template <class T>
__kernel void OCL_memset(__global T* buffer, const T value, const int size){
	int gdx = get_global_id(0);
	if(gdx < size){
		buffer[gdx] = value;	
	}
}

template __attribute__((mangled_name(oclmemfloat))) __kernel void OCL_memset(__global float* buffer, const float value, const int size);
template __attribute__((mangled_name(oclmemdouble))) __kernel void OCL_memset(__global double* buffer, const double value, const int size);

__kernel void OCL_memset2(__global int* buffer, const int value, const int size){
        int gdx = get_global_id(0);
        if(gdx < size){
                buffer[gdx] = value;    
        }
}

template <class T>
__kernel void im2col(const int n,__global T* data_im, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global T* data_col){
    int index=get_global_id(0);
    int tmp=get_global_size(0);
    for(index;index<n;index+=tmp){
        int w_out=index %width_col;
        index /= width_col;
        int h_out=index%height_col;
        int channel_in = index/height_col;
        int channel_out=channel_in *ksize *ksize;
        int h_in = h_out *stride-pad;
        int w_in = w_out *stride-pad;
        data_col +=(channel_out *height_col + h_out) *width_col + w_out;
        data_im +=(channel_in * height + h_in) *width + w_in;
        int i=0,j=0;
        for(i=0;i<ksize;++i){
            for(j=0;j<ksize;++j){
                int h = h_in+i;
                int w = w_in+j;
                if(h >= 0 && w >= 0 && h < height && w < width)
                    *data_col=data_im[i * width + j];
                else *data_col=0;
                data_col +=height_col *width_col;
            }
        }
    }
}

template __attribute__((mangled_name(im2colfloat))) __kernel void im2col(const int n,__global float* data_im, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global float* data_col); 
template __attribute__((mangled_name(im2coldouble))) __kernel void im2col(const int n,__global double* data_im, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global double* data_col); 


template <class T>
__kernel void MaxPoolForward(const int nthreads, __global T* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_size, const int stride, __global T* top_data){
     int index=get_global_id(0);
     int tmp=get_global_size(0);
     for(index;index<nthreads;index+=tmp){
         int pw = index % pooled_width;
         int ph = (index / pooled_width) % pooled_height;
         int c = (index / pooled_width / pooled_height) % channels;
         int n = index / pooled_width / pooled_height / channels;
         int hstart = ph * stride;
         int hend = min(hstart + kernel_size, height);
         int wstart = pw * stride;
         int wend = min(wstart + kernel_size, width);
         T maxval = -99999999;
         bottom_data += (n * channels + c) * height * width;
         for (int h = hstart; h < hend; ++h) {
           for (int w = wstart; w < wend; ++w) {
             maxval = max(maxval, bottom_data[h * width + w]);
           }   
         }
         top_data[index] = maxval;
     }

}
template __attribute__((mangled_name(MaxPoolForwardfloat))) __kernel void MaxPoolForward(const int nthreads, __global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_size, const int stride, __global float* top_data);
template __attribute__((mangled_name(MaxPoolForwarddouble))) __kernel void MaxPoolForward(const int nthreads, __global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width,  const int kernel_size, const int stride, __global double* top_data);


template <class T>
__kernel void AvePoolForward(const int nthreads, __global T* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_size, const int stride, const int pad, __global T* top_data){
    int index=get_global_id(0);
    int tmp=get_global_size(0);
    for(index;index<nthreads;index+=tmp){
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * stride - pad;
        int wstart = pw * stride - pad;
        int hend = min(hstart + kernel_size, height + pad);
        int wend = min(wstart + kernel_size, width + pad);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        T aveval = 0;
        bottom_data += (n * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            aveval += bottom_data[h * width + w];
          }
        }
        top_data[index] = aveval / pool_size;
    }

}
template __attribute__((mangled_name(AvePoolForwardfloat))) __kernel void AvePoolForward(const int nthreads, __global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_size, const int stride, const int pad, __global float* top_data);
template __attribute__((mangled_name(AvePoolForwarddouble))) __kernel void AvePoolForward(const int nthreads, __global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width,  const int kernel_size, const int stride, const int pad, __global double* top_data);

template <class T>
__kernel void MaxPoolBackward(const int nthreads, __global T* bottom_data, __global T* top_data, __global T* top_diff,
const int num, const int channels, const int height,
const int width, const int pooled_height, const int pooled_width,
const int kernel_size, const int stride, __global T* bottom_diff){
    int index = get_global_id(0);
    int total = get_global_size(0);
    for(index; index < nthreads; index += total){
        // find out the local index
        // find out the local offset
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;
        int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
        int phend = min(h / stride + 1, pooled_height);
        int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
        int pwend = min(w / stride + 1, pooled_width);
        T gradient = 0;
        T bottom_datum =
            bottom_data[((n * channels + c) * height + h) * width + w];
        top_data += (n * channels + c) * pooled_height * pooled_width;
        top_diff += (n * channels + c) * pooled_height * pooled_width;
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                gradient += top_diff[ph * pooled_width + pw] *
                    (bottom_datum == top_data[ph * pooled_width + pw]);
            }
        }
        bottom_diff[index] = gradient;

    }

}
template __attribute__((mangled_name(MaxPoolBackwardfloat))) __kernel void MaxPoolBackward(const int nthreads, __global float* bottom_data, __global float* top_data, __global float* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_size, const int stride, __global float* bottom_diff);
template __attribute__((mangled_name(MaxPoolBackwarddouble))) __kernel void MaxPoolBackward(const int nthreads, __global double* bottom_data, __global double* top_data, __global double* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_size, const int stride, __global double* bottom_diff);


template <class T>
__kernel void AvePoolBackward(const int nthreads, __global T* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_size, const int stride, const int pad, __global T* bottom_diff){
     int index = get_global_id(0);
     int total = get_global_size(0);
     for(index; index < nthreads; index += total){
	    int w = index % width + pad;
	    int h = (index / width) % height + pad;
	    int c = (index / width / height) % channels;
	    int n = index / width / height / channels;
	    int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
	    int phend = min(h / stride + 1, pooled_height);
	    int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
	    int pwend = min(w / stride + 1, pooled_width);
	    T gradient = 0;
	    top_diff += (n * channels + c) * pooled_height * pooled_width;
	    for (int ph = phstart; ph < phend; ++ph) {
	      for (int pw = pwstart; pw < pwend; ++pw) {
		// figure out the pooling size
		int hstart = ph * stride - pad;
		int wstart = pw * stride - pad;
		int hend = min(hstart + kernel_size, height + pad);
		int wend = min(wstart + kernel_size, width + pad);
		int pool_size = (hend - hstart) * (wend - wstart);
           gradient += top_diff[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;

   }
}

template __attribute__((mangled_name(AvePoolBackwardfloat))) __kernel void AvePoolBackward(const int nthreads, __global float* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_size, const int stride, const int pad, __global float* bottom_diff);
template __attribute__((mangled_name(AvePoolBackwarddouble))) __kernel void AvePoolBackward(const int nthreads, __global double* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width,  const int kernel_size, const int stride, const int pad, __global double* bottom_diff);

template <class T>
__kernel void ReLUForward(const int count, __global T* in, __global T* out){
	int index = get_global_id(0);
	int total = get_global_size(0);
	for(index; index < count; index += total)
		out[index] = in[index] > 0? in[index]:0;
}

template __attribute__ ((mangled_name(ReLUForwardfloat))) __kernel void ReLUForward(const int count, __global float* in, __global float* out);
template __attribute__ ((mangled_name(ReLUForwarddouble))) __kernel void ReLUForward(const int count, __global double* in, __global double* out);

template <class T>
__kernel void ReLUBackward(const int count, __global T* in_diff, __global T* in_data,__global T* out_diff){
	int index = get_global_id(0);
	int total = get_global_size(0);
	for(index; index < count; index += total)
		out_diff[index] = in_diff[index] *(in_data[index] > 0);
}

template __attribute__ ((mangled_name(ReLUBackwardfloat))) __kernel void ReLUBackward(const int count, __global float* in_diff, __global float* in_data, __global float* out_diff);
template __attribute__ ((mangled_name(ReLUBackwarddouble))) __kernel void ReLUBackward(const int count, __global double* in_diff, __global double* in_data, __global double* out_diff);

template <class T>
__kernel void get_max(const int num, const int dim, __global T* data, __global T* out){
     int index = get_global_id(0);
     int total = get_global_size(0);
     for(index; index < num; index +=  total){
	T maxval = -FLT_MAX;
        for (int i = 0; i <  dim; i++)
	maxval = max( data[index*dim + i], maxval );
        out[index] = maxval;
	}
}

template __attribute__ ((mangled_name(get_max_float))) __kernel void get_max(const int num, const int dim, __global float* data, __global float* out);
template __attribute__ ((mangled_name(get_max_double))) __kernel void get_max(const int num, const int dim, __global double* data, __global double* out);

template <class T>
__kernel void exp (const int num, __global T* data, __global T* out){
        int index = get_global_id(0);
        int total = get_global_size(0);
        for(index; index < num; index +=  total){
        out[index] = exp(data[index]);
        }
}

template __attribute__ ((mangled_name(exp_float))) __kernel void exp (const int num, __global float* data, __global float* out);
template __attribute__ ((mangled_name(exp_double))) __kernel void exp (const int num, __global double* data, __global double* out);

template <class T>
__kernel void softmax_div (const int num, const int dim, __global T* scale, __global T* data){
        int index = get_global_id(0);
        int total = get_global_size(0);
        for(index; index < num*dim; index +=  total){
        int n = index / dim;
        data[index] /= scale[n];
        }
}

template __attribute__ ((mangled_name(softmax_div_float))) __kernel void softmax_div (const int num, const int dim, __global float* scale, __global float* data);
template __attribute__ ((mangled_name(softmax_div_double))) __kernel void softmax_div (const int num, const int dim, __global double* scale, __global double* data);



template <class T>
__kernel void diff (const int num, const int dim, __global T* data, __global T* label){
        int index = get_global_id(0);
        int total = get_global_size(0);
        int offset;
	for(index; index < num; index +=  total){
  	offset = (int) label[index];
        data[index * dim + offset] -= 1;
        }
}

template __attribute__ ((mangled_name(diff_float))) __kernel void diff (const int num, const int dim, __global float* data, __global float* label);
template __attribute__ ((mangled_name(diff_double))) __kernel void diff (const int num, const int dim, __global double* data, __global double* label);

template <class T>
__kernel void scal (const int num, const T alpha, __global T* data){
        int index = get_global_id(0);
        int total = get_global_size(0);
        for(index; index < num; index +=  total){
        data[index] = data[index] * alpha;
        }
}

template __attribute__ ((mangled_name(scal_float))) __kernel void scal (const int num, const float alpha,  __global float* data);
template __attribute__ ((mangled_name(scal_double))) __kernel void scal (const int num, const double alpha,  __global double* data);

