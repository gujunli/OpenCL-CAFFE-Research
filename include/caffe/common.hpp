// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_
#include <CL/cl_ext.h>
#include <boost/shared_ptr.hpp>
#include <clAmdBlas.h>
//#include <driver_types.h>  // cuda driver types
#include <glog/logging.h>
#include "caffe/device.hpp"
// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// OpenCL: various checks for different function calls.

#define use_sgemm_ex
#define use_cpu_generator_dropout
//#define pipeline

#define OCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    CHECK_EQ(error, CL_SUCCESS) << " " << error; \
    if(CL_SUCCESS != error){ \
       LOG(INFO) << "failed";\
    } \
  } while (0)

#define AMDBLAS_CHECK(flag) \
  do { \
     cl_int error = flag; \
     CHECK_EQ(error, clAmdBlasSuccess) << " " << error; \
     if (error != clAmdBlasSuccess){ \
         LOG(INFO) << "AmdBlas Function Failed! Error Code:" << error; \
     } \
 } while(0)

//#define OCL_memset();

// Define not supported status for pre-6.0 compatibility.


namespace caffe {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;


// A singleton class to hold common caffe stuff, such as the handler that
class Caffe {
 public:
  ~Caffe();
  inline static Caffe& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new Caffe());
    }
    return *singleton_;
  }
  enum Brew { CPU, GPU };
  enum Phase { TRAIN, TEST };


  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // Returns the phase: TRAIN or TEST.
  inline static Phase phase() { return Get().phase_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the phase.
  inline static void set_phase(Phase phase) { Get().phase_ = phase; }
  static void set_random_seed(const unsigned int seed);
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();

 protected:
  shared_ptr<RNG> random_generator_;

  Brew mode_;
  Phase phase_;
  static shared_ptr<Caffe> singleton_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

inline int CAFFE_GET_BLOCKS(const int N) {
}


}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
