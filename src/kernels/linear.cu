# include<iostream>
# include "src/kernels/linear.h"
# include "src/utils/cuda_debug_utils.cuh"
/*
 * 矩阵乘Gemm的功能
 * Type1: 用于处理embedding vector
 *          Input shape = [sequence len, hidden_size]
 *          Weight shape = [hidden_size, hiddesn_size]
 * 
 * Type2: 用于批量处理最后预测目标词的softmax之前的linear: [batch_size,hidden_size]
 *          Input shape = [batch_size, hidden_size]
 *          Weight shape = [vocabulary_size, hidden_size]
 *          需要注意的是，因为weightshape是这么存的，因此计算时应该转置: Input*Weight^T
 *       
 * 
 * 
 * BatchGemm的功能:
 * Type1: 注意力分数Q*K计算,注意因为multi-head attention, headnums * headsize = hidden_size
 *          q shape: [batch_size, head nums, sequence len, head size]
 *          k shape: [batch_size, head_nums, sequence len, head size]
 *          此处k也需要转置，根据多维矩阵乘法规则，代表矩阵的最后两个维度[sequence len, head size]应该转置，其余代表矩阵所在的批次，如果缺少应当应用广播机制
 *          q * k = [batch_size, head nums, sequence len, head size] * [batch_size, head_nums, head size, sequence_len]
 *                = [batch size, head nums, sequence len, sequence len]
 *          
 *          更详细的解释下，QK的乘积是在计算每个头中每个位置的query和所有key向量的相似度，同时multi-head的划分是按照将embedding hidden size拆分划分的，这意味着对于每个head，它计算的是sequence中每个位置的query的部分特征和所有部分key向量的相似度，因此，每个head的QK矩阵应该是[seqlen,seqlen]
 *                  
 * Type2: 注意力权重QK*V计算
 *          qk shape: [batch size, head nums, sequence len, sequence len]
 *          v shape:  [batch size, head nums, sequence len, head size]
 * 
 *          此处就不需要转置
 * 
 * 
 * // 默认A = [m,k],B = [k,n],C = [m,n]
*/

/*
 * 可能涉及到的矩阵乘有如下
 * 
 * context decoder 
 * 
 * 1. q,k,v linear: 
 * 
 *          [num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits] = [num_tokens, head_num, head_size]
 * 
 * 2. attention output linear:
 * 
 *          [num_tokens, head_num, head_size] * [qhiddenunits, qhiddenunits] = [num_tokens, qhiddenunits]
 * 
 * self decoder
 * 
 * 1. q,k,v linear:
 *          
 *          [bs, qhiddenunits] * [qhiddenunits, hiddenunits] = [bs, head_num, head_size]
 * 
 * 2. attention output linear:
 *  
 *          [bs, qhiddenunits] * [qhiddenunits, qhiddenunits] = [bs, qhiddenunits]
 * 
 * 
 * LMHEAD linear
 * 
 *          [bs, qhiddenunits] * [vocabsize, qhiddenunits], transb
 * 
 * Gate:
 * 
 *          [bs/token_nums, qhiddenunits] * [qhiddenunits, intersize] = [bs/token_num, intersize]
 * 
 * UP:
 * 
 *          [bs/token_nums, qhiddenunits] * [qhiddenunits, intersize] = [bs/tokenn_nums, intersize]
 * 
 * 
 * FusedGateUpGemm
 * 
 *          [bs/token_nums, qhiddenunits] * [qhiddenunits, 2*intersize] = [bs/token_nums, 2*intersize](内存中等同于[bs/token_nums, 2, intersize])
 * 
 * DOWN
 * 
 *          [bs/token_nums, intersize] * [qhiddenunits, intersize] = [bs/token_nums, qhiddenunits]
*/






// 这里的写法目前只CHECK了矩阵B会转置，矩阵A默认不会转置
/*
    ***********************************************
    注意，cublas传入ld时，这个ld是与当前的trans无关的，ld是存储的矩阵本身的一个属性

    而m,n,k才是与trans有关的

    此处，一种应用中，理想的是[bs,hidden]*[hidden,vocabulary]
    最后获得bs个[1*vocabulary]去进行softmax，获得batchsize个词
*/
template<typename T>
__device__ void launchLinearGemm(   TensorWrapper<T>* input, 
                                    BaseWeight<T>& weight,
                                    TensorWrapper<T>* output, 
                                    cublasWrapper* cublas_wrapper,//cublas的功能集合
                                    bool trans_a,
                                    bool trans_b)
{
    LLM_CHECK_WITH_INFO(trans_a == true, "Attemp to trans matrix A, which situation is not considered ");

    cublasOperation_t transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;


    // CHECK
    if(!trans_a && trans_b){
        // CHECK [bs,1,hiddenunits] * [vocabulary, hiddenunits]的写法
        if(input->shape.size() == 3){
            LLM_CHECK_WITH_INFO(input->shape[2] == weight.shape[1],"while trans_b and input dimension is 3, the dimension is wrong:input->shape[2] != weight.shape[1]");
        }else{
            // [batch_size, hidden_units] * [vocabulary, hiddenunits]
            LLM_CHECK_WITH_INFO(input->shape[1] == weight.shape[1],"while trans_b, second dimension of B must be equal to second dimension of second dimension");
        }
    }

    // 仅考虑规范传入,暂时不考虑[batch_size,1,hiddenunits]的情况
    int m = trans_a ? input->shape[1] : input->shape[0];
    int k = trans_a ? input->shape[0] : input->shape[1];
    int n = trans_b ? weight.shape[0] : weight.shape[1];
    int lda = m;
    int ldb = k;
    int ldc = m;

    cublas_wrapper->Gemm(
        transA,
        transB,
        m,
        n,
        k,
        input->data,
        lda,
        weight.data,
        ldb,
        output->data,
        ldc,
        1.0f,
        0.0f
    );
}

// 仅考虑的是四维向量[a,b,c,d]*[a,b,d,c]
template<typename T>
void launchLinearStrideBatchGemm(   TensorWrapper<T>* input1, 
                                    TensorWrapper<T>* input2, 
                                    TensorWrapper<T>* output,
                                    cublasWrapper* cublas_wrapper,
                                    bool trans_a,
                                    bool trans_b)
{
    cublasOperation_t transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int m = trans_a ? input1->shape[3] : input1->shape[2];
    int k = trans_a ? input1->shape[2] : input1->shape[3];
    int n = trans_b ? input2->shape[2] : input2->shape[3];
    int lda = m;
    int ldb = n;
    int ldc = m;
    int64_t strideA = m*k;
    int64_t strideB = k*n;
    int64_t strideC = m*n;
    /*
        此处未添加
        1. 前两维审查机制
        2. 多维的拓展（优先级低）
        3. 前两维不同时的broadcast
    
    */
    int batchCount = input1->shape[0] * input1->shape[1];


    cublas_wrapper->strideBatchedGemm(
        transA,
        transB,
        m,
        n,
        k,
        input1->data,
        lda,
        strideA,
        input2->data,
        ldb,
        strideB,
        output->data,
        ldc,
        strideC,
        batchCount,
        1.0f,
        0.0f
    );
}                    