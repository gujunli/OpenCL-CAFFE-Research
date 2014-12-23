#include "caffe/device.hpp"
#include <stdio.h>
#include <fstream>
#include <iostream>
namespace caffe {

Device amdDevice;
char* buildOption = "-x clc++ ";


Device::~Device(){
    //clAmdBlasTeardown(); 
     clReleaseProgram(Program);
     clReleaseCommandQueue(CommandQueue);
     clReleaseContext(Context);
     LOG(INFO) << "device destructor";
}

cl_int Device::Init(){
    // we use the same random seed for the gaussian filler
    //srand(37);
    //Get Platform Infomation
    clGetPlatformIDs(0, NULL, &NumPlatforms);
    cl_platform_id PlatformIDs[NumPlatforms];
    clGetPlatformIDs(NumPlatforms, PlatformIDs, NULL);
    
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

    //Get Device Information
    clGetDeviceIDs(PlatformIDs[0], CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
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
    fprintf(stderr,"Err: Failed to open cl file!\n");
    return -1;
}

}  // namespace caffe

