# CMake file for CUDA objects

cmake_minimum_required(VERSION 3.0.0)
project(CudaKernel)

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif()

include_directories(
    basic/
    helpers/
    kernels/
    kernelsOverset/
)

file(GLOB_RECURSE CURRENT_HEADERS 
    basic/*.hpp
    basic/*.cuh
    # kernels/*.hpp
    # kernels/*.cuh
    kernelsOverset/*.hpp
    kernelsOverset/*.cuh
)

file(GLOB_RECURSE CURRENT_SOURCES 
    basic/*.cpp
    basic/*.cu
    # kernels/*.cpp
    # kernels/*.cu
    kernelsOverset/*.cpp
    kernelsOverset/*.cu
)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

cuda_add_library(${PROJECT_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
