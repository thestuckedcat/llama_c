cmake_minimum_required(VERSION 3.8)

project(llama LANGUAGES CXX CUDA)

enable_language(CUDA)
# 支持里面的文件使用类似#include"src/utils/tensor.h"
include_directories(${CMAKE_SOURCE_DIR})
add_subdirectory(src)
add_subdirectory(tests)