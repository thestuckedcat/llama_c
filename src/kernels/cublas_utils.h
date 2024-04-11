#pragma once
# include<cublasLt.h>
# include<cublas_v2.h>
# include<cuda_runtime.h>
# include<map>
# include<string>
# include"src/utils/macro.h"

/*
    cublaswrapper代表了其他我们可能用到的方法，参数等，主要参考Nvidia cublas

    docs.nvidia.cn/cuda/cublas/index.html#cublasgemmstridebatchedex
    2.8.13 cublasGemmBatchedEx()
    2.8.14 cublasGemmStrideBatchedEx()

    主要的就是包装Gemm，BatchGemm的各种类型下的方法
*/

class cublasWrapper{

private:
    cublasHandle_t cublas_handle_;
    /*
        The cudaDataType_t type is an enumerant to specify the data precision. It is used when the data reference does not carry the type itself (e.g void *)
    */
    cudaDataType_t Atype;
    cudaDataType_t Btype;
    cudaDataType_t Ctype;
    cublasComputeType_t computeType;
public:
    cublasWrapper(cublasHandle_t cublas_handle);
    ~cublasWrapper();

    //设置计算使用的数据类型
    void setFP32GemmConfig();
    void setFP16GemmConfig();

    // Gemm接口
    void Gemm(  cublasOperation_t transa,
                cublasOperation_t transb,
                const int m,
                const int n,
                const int k,
                const void* A,
                const int lda,
                const void* B,
                const int ldb,
                void* C,
                const int ldc,
                float alpha,
                float beta);


    // BatchGemm接口
    void strideBatchedGemm( cublasOperation_t transa,
                            cublasOperation_t transb,
                            const int m,
                            const int n,
                            const int k,
                            const void* A,
                            const int lda,
                            const int64_t strideA,
                            const void* B,
                            const int ldb,
                            const int64_t strideB,
                            void* C,
                            const int ldc,
                            const int64_t strideC,
                            const int batchCount,
                            float f_alpha,
                            float f_beta);


    // Transpose接口
    void cublasWrapper::Transpose(  int m,
                                int n,
                                float* d_C// Matrix to be transpose
                                );

};