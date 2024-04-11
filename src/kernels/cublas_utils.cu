# include "src/kernels/cublas_utils.h"
# include <iostream>
# include "src/utils/macro.h"


cublasWrapper::cublasWrapper(cublasHandle_t cublas_handle)
    :cublas_handle_(cublas_handle)
{}

cublasWrapper::~cublasWrapper(){

}

void cublasWrapper::setFP32GemmConfig(){
    Atype = CUDA_R_32F;
    Btype = CUDA_R_32F;
    Ctype = CUDA_R_32F;
    computeType = CUBLAS_COMPUTE_32F;
}
void cublasWrapper::setFP16GemmConfig(){
    Atype = CUDA_R_16F;
    Btype = CUDA_R_16F;
    Ctype = CUDA_R_16F;

    computeType = CUBLAS_COMPUTE_16F;
}


// 默认A = [m,k],B = [k,n],C = [m,n],注意传入m,n,k的时候必须是考虑了trans的实际可直接执行的矩阵乘法的mnk
// 调用cublasGemmStrideBatchedEx
// CUDA的GEMM可以理解为 C = α*A*B + β*C
// <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
void cublasWrapper::Gemm(  cublasOperation_t transa,
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
                float f_alpha = 1.0f,
                float f_beta = 0.0f)
{
    // 转换alpha,beta为对应的类型

    bool is_fp16_computeType = computeType == CUDA_R_16F ? true : false;
    half h_alpha = (half)(f_alpha);
    half h_beta = (half)(f_beta);

        //获取对应类型的布局通用指针（通用指针相当于wrapper）
    const void* alpha = is_fp16_computeType ? 
                        reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta = is_fp16_computeType ?
                        reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&h_beta);

    CHECK_CUBLAS(cublasGemmEx(
    cublas_handle_,  // context handle: 用于管理cuBLAS库上下文的句柄。
    transa,          // matrix trans operation for A: 指定矩阵A的转置操作，可以是CUBLAS_OP_N（不转置），CUBLAS_OP_T（转置）或CUBLAS_OP_C（共轭转置）。
    transb,          // matrix trans operation for B: 指定矩阵B的转置操作，同上。
    m,               // A = [m,k],B = [k,n],C = [m,n]: 指定矩阵A的行数和C的行数。
    n,               // 指定矩阵B的列数和C的列数。
    k,               // 指定矩阵A的列数和B的行数。
    alpha,           // 缩放因子，用于乘以矩阵A和B乘积的结果。
    A,               // 指向矩阵A的指针。
    Atype,           // 矩阵A的数据类型，例如CUBLAS_DATA_FLOAT、CUBLAS_DATA_DOUBLE等。
    lda,             // A的领先维度（leading dimension），在不转置时是A的行数，在转置时是A的列数。
    B,               // 指向矩阵B的指针。
    Btype,           // 矩阵B的数据类型，同Atype。
    ldb,             // B的领先维度，处理逻辑同lda。
    beta,            // 缩放因子，用于乘以矩阵C的原始内容。
    C,               // 指向矩阵C的指针，存放计算结果。
    Ctype,           // 矩阵C的数据类型，同Atype和Btype。
    ldc,             // C的领先维度，通常是C的行数。
    computeType,     // 计算使用的数据类型，影响计算精度和性能。
    CUBLAS_GEMM_DEFAULT // 指定GEMM的算法策略，CUBLAS_GEMM_DEFAULT表示使用默认算法。
    ));         
    
}



// 26:56
void cublasWrapper::strideBatchedGemm(  cublasOperation_t transa,
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
                                        float f_alpha = 1.0f,
                                        float f_beta = 0.0f)
{
    bool is_fp16_computeType = computeType == CUDA_R_16F ? true:false;

    half h_alpha = (half) f_alpha;
    half h_beta = (half) f_beta;

    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*> (&f_alpha);

    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) :reinterpret_cast<void*>(&f_beta);

    /*
        在batchedGemm中，通常将矩阵乘法分成两部分，一部分描述矩阵batch，一部分描述矩阵本身
        以两个四维矩阵A:[a,b,c,d] B:[a,b,d,c]为例，可以看成a*b个c*d和d*c的矩阵乘法

        因此，我们首先将四维矩阵向量化，然后用stride来描述每次定位到下一个矩阵切片的步长

        strideA = c*d           strideB = d*c
        lda     = c             ldb     = d
        batchCount = a * b

        这里需要注意的一点是A*B的传入部分
        理论上
        A:[m,n] B:[k,n] transb = ON
        比
        A:[m,n] B:[n,k] transb = OFF
        更加高效，因为列主序保证了第一种情况能内存连续的读取m*1和1*k向量计算
        但是，考虑到cublas优化，实际上一般相差不会很大
    */
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        cublas_handle_,
        transa,
        transb,

        m,
        n,
        k,

        alpha,
        A,
        Atype,
        lda,
        strideA,
        B,
        Btype,
        ldb,
        strideB,

        beta,
        C,
        Ctype,
        ldc,
        strideC,
        
        batchCount,
        computeType,
        CUBLAS_GEMM_DEFAULT
    ));
}   



// Used to transpos matrix A
void cublasWrapper::Transpose(  int m,
                                int n,
                                float* d_C// Matrix to be transpose
                                )
{
    /*
        C = α op(A) + β op(B)

        lda,ldb affected by trans

        m: number of rows of matrix op(A) and C.

        n: number of columns of matrix op(B) and C.


    */
    float alpha = 1.0f;
    float beta = 0.0f;
    int ldc_t = n;

    float* Newresult;
    CHECK(cudaMalloc((void**)&Newresult, sizeof(float) * m*n));
    cublasStatus_t status = cublasSgeam(cublas_handle_,
                                     CUBLAS_OP_T, // 操作A，转置
                                     CUBLAS_OP_N, // 操作B，不变，因为B不被用到
                                     n,           // 矩阵C'的行数（原C的列数）
                                     m,           // 矩阵C'的列数（原C的行数）
                                     &alpha,      // A的乘数系数
                                     d_C,         // 原矩阵C
                                     m,           // 原C的领先维度
                                     &beta,       // B的乘数系数，设置为0因为B不被使用
                                     NULL,        // B，未使用
                                     ldc_t,       // B的领先维度，未使用
                                     Newresult, // 结果矩阵C'
                                     ldc_t);      // 结果矩阵C'的领先维度

    CHECK(cudaMemcpy(d_C, Newresult, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
    cudaFree(Newresult);
}



