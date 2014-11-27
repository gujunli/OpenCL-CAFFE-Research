// Copyright 2014 BVLC and contributors.

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <cublas_v2.h>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
static const clAmdBlasOrder order = clAmdBlasColumnMajor;
#define pi 3.1415926

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
    clAmdBlasTranspose transA = (TransA == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    clAmdBlasTranspose transB = (TransB == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    int ldc = N;
    AMDBLAS_CHECK( clAmdBlasSgemm(amdDevice.col, transB, transA, N, M, K, (cl_float)alpha, (cl_mem)B, ldb, (cl_mem)A, lda, (cl_float)beta, (cl_mem)C, ldc, 1, &(amdDevice.CommandQueue), 0, NULL, NULL) );
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
    clAmdBlasTranspose transA = (TransA == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    clAmdBlasTranspose transB = (TransB == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    int ldc = N;
    AMDBLAS_CHECK( clAmdBlasDgemm(amdDevice.col, transB, transA, N, M, K, (cl_float)alpha, (cl_mem)B, ldb, (cl_mem)A, lda, (cl_float)beta, (cl_mem)C, ldc, 1, &(amdDevice.CommandQueue), 0, NULL, NULL) );
}

template <>
void caffe_gpu_gemm_ex<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A,const int offA, const float* B, const int offB, const float beta, float* C, const int offC) {
  // Note that cublas follows fortran order.
    clAmdBlasTranspose transA = (TransA == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    clAmdBlasTranspose transB = (TransB == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    int ldc = N;
    AMDBLAS_CHECK( clAmdBlasSgemmEx(amdDevice.col, transB, transA, N, M, K, (cl_float)alpha, (cl_mem)B, offB, ldb, (cl_mem)A, offA, lda, (cl_float)beta, (cl_mem)C, offC, ldc, 1, &(amdDevice.CommandQueue), 0, NULL, NULL) );
}

template <>
void caffe_gpu_gemm_ex<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A,const int offA, const double* B, const int offB, const double beta, double* C, const int offC) {
  // Note that cublas follows fortran order.
    clAmdBlasTranspose transA = (TransA == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    clAmdBlasTranspose transB = (TransB == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    int ldc = N;
    AMDBLAS_CHECK( clAmdBlasSgemmEx(amdDevice.col, transB, transA, N, M, K, (cl_float)alpha, (cl_mem)B, offB, ldb, (cl_mem)A, offA, lda, (cl_float)beta, (cl_mem)C, offC, ldc, 1, &(amdDevice.CommandQueue), 0, NULL, NULL) );
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_gpu_gemvv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, size_t offA, int lda, 
    const float* x, size_t offx, const float beta, int incx, 
    float* y, size_t offy, int incy) {
    clAmdBlasTranspose transA = (TransA == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    AMDBLAS_CHECK( clAmdBlasSgemvEx(amdDevice.row, transA,
                                  M, N, (cl_float)alpha, (cl_mem)A, offA, lda,
                                  (cl_mem)x, offx, incx, (cl_float)beta, 
                                  (cl_mem)y, offy, incy,
                                  1, &(amdDevice.CommandQueue), 0, NULL, NULL) );
}

template <>
void caffe_gpu_gemvv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, size_t offA, int lda,
    const double* x, size_t offx, const double beta, int incx,
    double* y, size_t offy, int incy) {
    clAmdBlasTranspose transA = (TransA == CblasNoTrans)? clAmdBlasNoTrans : clAmdBlasTrans;
    AMDBLAS_CHECK( clAmdBlasSgemvEx(amdDevice.row, transA, M, N, (cl_double)alpha, (cl_mem)A, offA, lda, (cl_mem)x, offx, incx, (cl_double)beta, (cl_mem)y, offy, incy, 1, &(amdDevice.CommandQueue), 0, NULL, NULL) );

}


template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
    AMDBLAS_CHECK( clAmdBlasSaxpy(N, alpha, (cl_mem)X, 0, 1, (cl_mem)Y, 0, 1, 1, &(amdDevice.CommandQueue),0, NULL, NULL) );
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
    AMDBLAS_CHECK( clAmdBlasDaxpy(N, alpha, (cl_mem)X, 0, 1, (cl_mem)Y, 0, 1, 1, &(amdDevice.CommandQueue),0, NULL, NULL) );
}

template <>
void caffe_set(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <>
void caffe_set(const int N, const double alpha, double* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(double) * N);
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_copy<float>(const int N, const float* X, float* Y) {
  cblas_scopy(N, X, 1, Y, 1);
}

template <>
void caffe_copy<double>(const int N, const double* X, double* Y) {
  cblas_dcopy(N, X, 1, Y, 1);
}

template <>
void caffe_gpu_copy<float>(const int N, const float* X, float* Y) {
  AMDBLAS_CHECK( clAmdBlasScopy( N, (cl_mem)X, 0,1, (cl_mem)Y, 0, 1, 1, &(amdDevice.CommandQueue), 0, NULL, NULL) );

}

template <>
void caffe_gpu_copy<double>(const int N, const double* X, double* Y) {
  AMDBLAS_CHECK( clAmdBlasDcopy( N, (cl_mem)X, 0,1, (cl_mem)Y, 0, 1, 1, &(amdDevice.CommandQueue), 0, NULL, NULL) );
}

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
   AMDBLAS_CHECK(clAmdBlasSscal(N, alpha, (cl_mem)X, 0, 1, 1, &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  AMDBLAS_CHECK(clAmdBlasDscal(N, alpha, (cl_mem)X, 0, 1, 1, &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }

  LOG(INFO) << "caffe_rng_uniform";
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
      //variate_generator(37, random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
  LOG(INFO) << "caffe_rng_guassian";
  /*
  //srand(37);
  for (int i = 0; i < n; ++i) {
    float u1 = ((float)rand())/RAND_MAX;
    float u2 = ((float)rand())/RAND_MAX;
    float x = sqrt(-2*log(u1))*sin(2*pi*u2);
    r[i] = a + sigma*x;
  }
   for(int i=0; i<5; i++)
   {
      LOG(INFO)<<a<<"\t"<<sigma << "   rrrr[" <<i <<"] =" << r[i] ; 
   }
*/
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
  LOG(INFO) << "caffe_rng_bernoulli";
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <>
float caffe_cpu_dot<float>(const int n, const float* x, const float* y) {
  return cblas_sdot(n, x, 1, y, 1);
}

template <>
double caffe_cpu_dot<double>(const int n, const double* x, const double* y) {
  return cblas_ddot(n, x, 1, y, 1);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
  //need to pass in scratchBuff
  //AMDBLAS_CHECK(clAmdBlasSdot(n, out, 0, x, 0, 1, y, 0, 1, scratch_buf, 1, &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
  //need to pass in scratchBuff
  //AMDBLAS_CHECK(clAmdBlasDdot(n, out, 0, x, 0, 1, y, 0, 1, scratch_buf, 1, &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
int caffe_cpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(static_cast<uint32_t>(x[i]) ^
                               static_cast<uint32_t>(y[i]));
  }
  return dist;
}

template <>
int caffe_cpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcountl(static_cast<uint64_t>(x[i]) ^
                                static_cast<uint64_t>(y[i]));
  }
  return dist;
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

INSTANTIATE_CAFFE_CPU_UNARY_FUNC(sign);
INSTANTIATE_CAFFE_CPU_UNARY_FUNC(sgnbit);
INSTANTIATE_CAFFE_CPU_UNARY_FUNC(fabs);

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

}  // namespace caffe
