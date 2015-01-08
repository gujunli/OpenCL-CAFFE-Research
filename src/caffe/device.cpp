#include "caffe/common.hpp"
#include "caffe/device.hpp"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <malloc.h>
namespace caffe {

Device amdDevice;
char* buildOption = "-x clc++ ";

Device::~Device(){
    //clAmdBlasTeardown(); 
     free((void*)platformIDs);
     free(DeviceIDs);
     clReleaseProgram(Program);
     clReleaseCommandQueue(CommandQueue);
     clReleaseContext(Context);
     LOG(INFO) << "device destructor";
}


cl_int Device::Init(){

    //Get Platform Infomation
    DisplayPlatformInfo();
  
    clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_platform_id PlatformIDs[numPlatforms];
    clGetPlatformIDs(numPlatforms, PlatformIDs, NULL);
    
    size_t nameLen;
    cl_int res = clGetPlatformInfo(PlatformIDs[0], CL_PLATFORM_NAME, 64, platformName, &nameLen);
    if(res != CL_SUCCESS){
        fprintf(stderr, "Err: Failed to Get Platform Info\n", res);
        return 0;
    }
    platformName[nameLen] = 0;

    //Get OpenCL Information 
    //res = clGetPlatformInfo(PlatformIDs[0], CL_PLATFORM_VERSION, 64, openclVersion, &nameLen);
    //if(res != CL_SUCCESS) {
    //    fprintf(stderr, "Err: Get OpenCL Info failed!\n", res);
    //    return 0;
    //}
    //openclVersion[nameLen] = 0;
    //printf("%s %s\n", platformName, openclVersion);
  
    GetDeviceInfo();
    
    //Get Device Information
    clGetDeviceIDs(PlatformIDs[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    cl_uint uiNumDevices = numDevices;
    cl_device_id * pDevices;
    if(0 == uiNumDevices){
        //clGetDeviceIDs(PlatformIDs[0],CL_DEVICE_TYPE_CPU,0,NULL,&uiNumDevices);
        if(0 == uiNumDevices){
            fprintf(stderr, "Err:There is no any CPU or GPU device\n");
            return 0;
        }
        pDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id));
        clGetDeviceIDs(PlatformIDs[0], CL_DEVICE_TYPE_CPU, uiNumDevices, pDevices, NULL);
    }
    else{
        pDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id));
        clGetDeviceIDs(PlatformIDs[0], CL_DEVICE_TYPE_GPU, uiNumDevices, pDevices, NULL);
    }

    //Create Context
    Context = clCreateContext(NULL, 1, pDevices, NULL, NULL, NULL);
    if(NULL == Context){
        fprintf(stderr,"Err: Failed to Create Context\n");
        return 0;
    }

    //Create CommandQueue
    CommandQueue = clCreateCommandQueue(Context, pDevices[0], 0, NULL);
    if(NULL == CommandQueue){
        fprintf(stderr,"Err: Failed to Create Commandqueue\n");
        return 0;
    }

    //Read our own kernel file
    const char *pFileName = "../../src/caffe/OCL_kernel.cl";
    const char *pSource;
    std::string strSource = "";
    ConvertToString(pFileName, strSource);
    pSource = strSource.c_str();
    size_t uiArrSourceSize[] = {0};
    uiArrSourceSize[0] = strlen(pSource);
    Program = NULL;
    Program = clCreateProgramWithSource(Context, 1, &pSource, uiArrSourceSize, NULL);
    if(NULL == Program){
        fprintf(stderr,"Err: Failed to create program\n");
    }

    //Build Program
    cl_int iStatus = clBuildProgram(Program, 1, pDevices, buildOption, NULL, NULL);
    LOG(INFO) << "Build Program";
    if(CL_SUCCESS!=iStatus){
        fprintf(stderr,"Err: Failed to build program\n");
        char szBuildLog[16384];
        clGetProgramBuildInfo(Program, *pDevices, CL_PROGRAM_BUILD_LOG, sizeof(szBuildLog), szBuildLog, NULL);
        std::cout << szBuildLog;
        clReleaseProgram(Program);
    }

    /*
    //Setup AmdBlas;
    cl_int err;
    err = clAmdBlasSetup();
    if(err != CL_SUCCESS){
        printf("clAmdBlasSetup() failed with %d\n", err);
    }
    */
    row = clAmdBlasRowMajor;
    col = clAmdBlasColumnMajor;
   
    return 0;
}


//Use to read OpenCL source code
cl_int Device::ConvertToString(const char *pFileName,std::string &Str){
    size_t uiSize=0;
    size_t uiFileSize=0;
    char *pStr=NULL;
    std::fstream fFile(pFileName,(std::fstream::in|std::fstream::binary));
    if(fFile.is_open()){
        fFile.seekg(0,std::fstream::end);
        uiSize=uiFileSize=(size_t)fFile.tellg();
        fFile.seekg(0,std::fstream::beg);
        pStr=new char[uiSize+1];

        if(NULL==pStr){
            fFile.close();
            return 0;
        }
        fFile.read(pStr,uiFileSize);
        fFile.close();
        pStr[uiSize]='\0';
        Str=pStr;
        delete[] pStr;
        return 0;
    }
    LOG(ERROR) << "Err: Failed to open cl file!";
    return -1;
}

void Device::DisplayPlatformInfo(){
   cl_int err;
   size_t size;

   err = clGetPlatformIDs (0, NULL, &numPlatforms);
   if(err != CL_SUCCESS || numPlatforms <=0)
   {
      LOG(ERROR) << "Failed to find any OpenCL platform.";
      return;
   }

   platformIDs = (cl_platform_id *) malloc (sizeof(cl_platform_id) * numPlatforms);
   err = clGetPlatformIDs (numPlatforms, platformIDs, NULL);
   if(err != CL_SUCCESS)
   {
      LOG(ERROR) << "Failed to find any OpenCL platform.";
      return;
   }

   LOG(INFO) << "Number of platforms found:" << numPlatforms;

  //iterate through the list of platforms displaying platform information
  for (cl_uint i = 0; i < numPlatforms; i++ ){
  DisplayInfo(platformIDs[i], CL_PLATFORM_NAME, "CL_PLATFORM_NAME");
  DisplayInfo(platformIDs[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
  DisplayInfo(platformIDs[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
  DisplayInfo(platformIDs[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
  DisplayInfo(platformIDs[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");
  }
   
}

void Device::DisplayInfo(cl_platform_id id, cl_platform_info name, std::string str){
    cl_int err;
    std::size_t paramValueSize;

    err = clGetPlatformInfo(id, name, 0, NULL, &paramValueSize);  
   if(err != CL_SUCCESS)
   {
      LOG(ERROR) << "Failed to find OpenCL platform:" << str;
      return;
   }
   
   char * info = (char *) alloca (sizeof(char) * paramValueSize);
   err = clGetPlatformInfo(id, name, paramValueSize, info, NULL);
   if(err != CL_SUCCESS)
   {
      LOG(ERROR) << "Failed to find OpenCL platform:" << str;
      return;
   }

   LOG(INFO) << "\t" << str << "\t" << info;
}

void Device::GetDeviceInfo(){
    cl_int err;
    //by default, we select the first platform. can be extended for more platforms
    //query GPU device for now
    err = clGetDeviceIDs(platformIDs[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    // we allow program run if no GPU is found. Just return. No error reported.
    if (numDevices < 1)
    {
      LOG(INFO) << "No GPU Devices found for platform" << platformIDs[0];
      LOG(WARNING) << "No GPU Devices found for platform" << platformIDs[0];
      return;
    }
    
    DeviceIDs = (cl_device_id *) malloc (sizeof(cl_device_id) * numDevices);
    err = clGetDeviceIDs(platformIDs[0], CL_DEVICE_TYPE_GPU, numDevices, DeviceIDs, NULL);
    if(err != CL_SUCCESS)
    {
      LOG(INFO) << "Failed to find any GPU devices.";
      return;
    }

    LOG(INFO) << "Number of devices found:" << numDevices;
    for(cl_uint i = 0; i < numDevices; i++){
    LOG(INFO) << "\t" << "DeviceID" << ":\t" <<DeviceIDs[i];
    DisplayDeviceInfo<cl_device_type>(DeviceIDs[i], CL_DEVICE_TYPE, "Device Type");
    DisplayDeviceInfo<cl_uint>(DeviceIDs[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, "Max clock frequency MHz");
    DisplayDeviceInfo<cl_bool>(DeviceIDs[i], CL_DEVICE_HOST_UNIFIED_MEMORY, "Host-Device unified mem");
    DisplayDeviceInfo<cl_bool>(DeviceIDs[i], CL_DEVICE_ERROR_CORRECTION_SUPPORT, "ECC support");
    DisplayDeviceInfo<cl_bool>(DeviceIDs[i], CL_DEVICE_ENDIAN_LITTLE, "Endian little");
    DisplayDeviceInfo<cl_uint>(DeviceIDs[i], CL_DEVICE_MAX_COMPUTE_UNITS, "Max compute units");
    DisplayDeviceInfo<size_t>(DeviceIDs[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, "Max work group size");
    DisplayDeviceInfo<cl_uint>(DeviceIDs[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "Max work item dimensions");
    DisplayDeviceInfo<size_t *>(DeviceIDs[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, "Max work item sizes");
    DisplayDeviceInfo<cl_command_queue_properties>(DeviceIDs[i], CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES");
    DisplayDeviceInfo<cl_device_exec_capabilities>(DeviceIDs[i], CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES");
    DisplayDeviceInfo<cl_ulong>(DeviceIDs[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, "Max mem alloc size");
    DisplayDeviceInfo<cl_ulong>(DeviceIDs[i], CL_DEVICE_GLOBAL_MEM_SIZE, "Global mem size");
    DisplayDeviceInfo<cl_ulong>(DeviceIDs[i], CL_DEVICE_LOCAL_MEM_SIZE, "Local mem size");
    }
    
    
}

template <typename T>
void Device::DisplayDeviceInfo(cl_device_id id, cl_device_info name, std::string str){
    cl_int err;
    std::size_t paramValueSize;

    err = clGetDeviceInfo(id, name, 0, NULL, &paramValueSize);  
   if(err != CL_SUCCESS)
   {
      LOG(ERROR) << "Failed to find OpenCL device info:" << str;
      return;
   }
  
   std::string content; 
   T * info = (T *) alloca (sizeof(T) * paramValueSize);
   err = clGetDeviceInfo(id, name, paramValueSize, info, NULL);
   if(err != CL_SUCCESS)
   {
      LOG(ERROR) << "Failed to find OpenCL device info:" << str;
      return;
   }


   switch(name)
{
    case CL_DEVICE_TYPE:
    {
        std::string deviceType;
        appendBitfield<cl_device_type>(
        *(reinterpret_cast<cl_device_type*>(info)),CL_DEVICE_TYPE_CPU,"CL_DEVICE_TYPE_CPU",deviceType);

        appendBitfield<cl_device_type>(
        *(reinterpret_cast<cl_device_type*>(info)),CL_DEVICE_TYPE_GPU,"CL_DEVICE_TYPE_GPU",deviceType);

        appendBitfield<cl_device_type>(
        *(reinterpret_cast < cl_device_type*>(info)),CL_DEVICE_TYPE_ACCELERATOR,"CL_DEVICE_TYPE_ACCELERATOR",deviceType);

        appendBitfield<cl_device_type>(
        *(reinterpret_cast < cl_device_type*>(info)),CL_DEVICE_TYPE_DEFAULT,"CL_DEVICE_TYPE_DEFAULT",deviceType);
        
	LOG(INFO) << "\t " << str << ":\t" << deviceType;
    }
        break;
    case CL_DEVICE_EXECUTION_CAPABILITIES:
    {
        std::string memType;
        appendBitfield<cl_device_exec_capabilities>(
        *(reinterpret_cast<cl_device_exec_capabilities*>(info)),CL_EXEC_KERNEL,"CL_EXEC_KERNEL",memType);

        appendBitfield<cl_device_exec_capabilities>(
        *(reinterpret_cast<cl_device_exec_capabilities*>(info)),CL_EXEC_NATIVE_KERNEL,"CL_EXEC_NATIVE_KERNEL",memType);

        LOG(INFO) << "\t " << str << ":\t" << memType;

    }
       break;
    case CL_DEVICE_QUEUE_PROPERTIES:
        {
            std::string memType;
            appendBitfield<cl_device_exec_capabilities>(*(reinterpret_cast<cl_device_exec_capabilities*>(info)),CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,"CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE",memType);

            appendBitfield<cl_device_exec_capabilities>(*(reinterpret_cast<cl_device_exec_capabilities*>(info)),CL_QUEUE_PROFILING_ENABLE,"CL_QUEUE_PROFILING_ENABLE",memType);

            LOG(INFO) << "\t " << str << ":\t" << memType;
        }
        break;
    default:
        LOG(INFO) << "\t" << str << ":\t" << *info;
        break;
}

}

template<typename T>
void Device::appendBitfield(T info, T value , std::string name , std::string &str)
{
    if(info & value)
    {
        if (str.length() > 0)
        {
            str.append(" | ");
        }
        str.append(name);
    }
}


}  // namespace caffe

