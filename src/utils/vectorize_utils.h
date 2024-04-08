#pragma once

#include <cuda.h>
#include <cuda_fp16.h> //half

// 将对应的给定类型T自动转化为CUDA支持的对应向量类型


template<typename T>
struct Vec{
    // 为类型T创建类型别名
    using type=T;
    // 向量长度
    static constexpr int size = 0;
};


//显式实例化
template <>
struct Vec<float>{
    using type = float4;
    static constexpr int size = 4;
};

// 这里使用half2而不是half8是考虑到了硬件直接支持的情况，half8可能需要更多操作，而half2可能被直接支持。
template<>
struct Vec<half>{
    using type = half2;
    static constexpr int size = 2;
};


