#ifndef CAFFE_DEVICE_HPP
#define CAFFE_DEVICE_HPP
#include <CL/cl.h>
#include <string>
#include <fstream>
#include "caffe/common.hpp"
class Device{
public:
    Device():NumPlatforms(0),uiNumDevices(0){}
    cl_uint NumPlatforms;
    char platformName[64];
    char openclVersion[64];
    cl_uint uiNumDevices;
    cl_device_id *pDevices;
    cl_context Context;
    cl_command_queue CommandQueue;
    cl_program Program; 
    clAmdBlasOrder order;

     
    cl_int Init(); 
    cl_int ConvertToString(const char *pFileName,std::string &Str);

};
extern char* buildOption;
extern Device amdDevice;
#endif
