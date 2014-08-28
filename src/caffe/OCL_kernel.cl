

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
__kernel void OCL_memset(__global T* buffer, const T value, const int size){
	int gdx = get_global_id(0);
	if(gdx < size){
		buffer[gdx] = value;	
	}
}

template __attribute__((mangled_name(oclmemfloat))) __kernel void OCL_memset(__global float* buffer, const float value, const int size);
template __attribute__((mangled_name(oclmemdouble))) __kernel void OCL_memset(__global double* buffer, const double value, const int size);



