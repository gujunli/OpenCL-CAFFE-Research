// Copyright 2014 BVLC and contributors.

#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;
// seeding
int64_t cluster_seedgen(void) {
  //To fix: for now we use fixed seed to get same result each time
  /*
  int64_t s, seed, pid;
  pid = getpid();
  s = time(NULL);
  seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
  */
  LOG(WARNING) << "return fixed seed 37";
  return 37;
}



Caffe::Caffe()
    : mode_(Caffe::CPU), phase_(Caffe::TRAIN),
      random_generator_() {

    //For debugging: in order to have deterministic results, we use the same random seed for the gaussian filler
    //need to be removed for real training
    //srand(37);

    cl_int err;
    err = clblasSetup();
    if (err != CL_SUCCESS) {
       LOG(ERROR) << "clAmdBlasSetup() failed with " << err;
    }
    

  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
}

Caffe::~Caffe() {
  clblasTeardown();
}

void Caffe::set_random_seed(const unsigned int seed) {
  // Yangqing's note: simply setting the generator seed does not seem to
  // work on the tesla K20s, so I wrote the ugly reset thing below.
  LOG(WARNING) << "set_random_seed";
}

void Caffe::SetDevice(const int device_id) {
}

void Caffe::DeviceQuery() {
}


class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

}  // namespace caffe
