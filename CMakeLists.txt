CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(cudaTest)

FIND_PACKAGE(CUDA REQUIRED)

SET(EXECUTABLE_OUTPUT_PATH /home/srpowell/tc/software/build/cudaTest/bin)
SET(LIBRARY_OUTPUT_PATH /home/srpowell/tc/software/build/cudaTest/lib)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} main.cpp gpuintadd.cuh gpuintadd.cu cpuadd.cpp cpuadd.h gpuadd.cuh gpuadd.cu )
#TARGET_LINK_LIBRARIES


#ADD_EXECUTABLE()
