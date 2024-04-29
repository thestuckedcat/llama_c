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


/*
    ***********************************************
    传入CUBLAS时，我们需要理清几点
    m,n,k：代表trans之后的op(A) * op(B)时，应该是[m,k]*[k,n]
    lda,ldb,ldc：代表trans之后op(A)的行数，op(B)的行数



    此处，一种应用中，理想的是[bs,hidden]*[hidden,vocabulary]
    最后获得bs个[1*vocabulary]去进行softmax，获得batchsize个词
*/
template<typename T>
__device__ void launchLinearGemm(   TensorWrapper<T>* input1, 
                                    TensorWrapper<T>* input2,
                                    TensorWrapper<T>* output, 
                                    cublasWrapper* cublas_wrapper,//cublas的功能集合
                                    bool trans_a,
                                    bool trans_b,
                                    bool trans_c)
{
    /*
        使用该函数时，使用最终矩阵，这意味着矩阵已经完成了列主序的构造，否则使用trans来模拟列主序
        
        例如，对于行主序的C = A * B^T,
        这意味着对于原本的计算,trans_a = false, trans_b = true
        因为是行主序输入，因此trans_c=true，代表我们实际上计算时应该为
        C^T = B * A^T
        
        这意味着，输入时，你的输入为
        input1 = B, input2 = A, trans_a = false, trans_b = true, trans_c = true
    */

    // input1为左矩阵，input2为右矩阵，应该考虑转置，并获得一个行主序的output
    
    /*

        处理两种shape

        第一种是传统的二维矩阵相乘

        第二种是在自回归生成时，计算添加进KV cache的vector
        [bs, 1, hiddenunits] * [vocabulary, hiddenunits]
        这种方式下，实际上data也是可以被看做[bs,hidden_units]的组织形式的


        同时，这里考虑如果trans_c = true,那么本来的

        Output = Tensor * Weight

        就变成了

        Output^T = Weight^T * Tensor^T

        在这种情况下，应该额外将output数据转置，以获得行主序的data。
    */
    cublasOperation_t transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    // CHECK
    int input1_1 = -1;
    int input1_2 = -1;
    int input2_1 = -1;
    int input2_2 = -1;
    if(input2->shape.size() > 2 || input1->shape.size() > 2){
        //CHECK TYPE2
        LLM_CHECK_WITH_INFO(input2->shape.size() <= 3 && input1->shape.size() <= 3, 
                            "Wrongly use Gemm with four or more dimension matrix");
        if(input1->shape.size() == 3){
            input1_1 = input1->shape[0];
            input1_2 = input1->shape[2];
        }else if(input1->shape.size() == 2){
            input1_1 = input1->shape[0];
            input1_2 = input1->shape[1];
        }

        if(input2->shape.size() == 3){
            input2_1 = input1->shape[0];
            input2_2 = input1->shape[2];
        }else if(input2->shape.size() == 2){
            input2_1 = input1->shape[0];
            input2_2 = input1->shape[1];
        }

        LLM_CHECK_WITH_INFO(input1_1 > 0 && 
                            input1_2 > 0 &&
                            input2_1 > 0 &&
                            input2_2 > 0,
                            "Some thing wrong with SHAPE while check Gemm Type 1");

        if(trans_a && trans_b){
            LLM_CHECK_WITH_INFO(input1_1 == input2_2, 
                                "All trans and shape wrong");
        }else if(trans_a && !trans_b){
            LLM_CHECK_WITH_INFO(input1_1 == input2_1, 
                                "trans_a and shape wrong");
        }else if(!trans_a && trans_b){
            LLM_CHECK_WITH_INFO(input1_2 == input2_2, 
                                "trans_b and shape wrong");
        }else{
            //!trans_a && !trans_b
            LLM_CHECK_WITH_INFO(input1_2 == input2_1, 
                                "No trans and shape wrong");
        }

    }
    else if(input2->shape.size() == 2 && input1->shape.size() == 2)
    {
        // CHECK TYPE1
        input1_1 = input1->shape[0];
        input1_2 = input1->shape[1];
        input2_1 = input2->shape[0];
        input2_2 = input2->shape[1];

        LLM_CHECK_WITH_INFO(input1_1 > 0 && 
                            input1_2 > 0 &&
                            input2_1 > 0 &&
                            input2_2 > 0,
                            "Some thing wrong with SHAPE while check Gemm Type 1");

        if(trans_a && trans_b){
            LLM_CHECK_WITH_INFO(input1_1 == input2_2, 
                                "All trans and shape wrong");
        }else if(trans_a && !trans_b){
            LLM_CHECK_WITH_INFO(input1_1 == input2_1, 
                                "trans_a and shape wrong");
        }else if(!trans_a && trans_b){
            LLM_CHECK_WITH_INFO(input1_2 == input2_2, 
                                "trans_b and shape wrong");
        }else{
            //!trans_a && !trans_b
            LLM_CHECK_WITH_INFO(input1_2 == input2_1, 
                                "No trans and shape wrong");
        }
        
    }else{
        LLM_CHECK_WITH_INFO(false,
                            "The input shape is neither the two type in GEMM, wrong call");
    }


    //Set m,k,n,考虑到output实际上没有数据，所以不管
    int m = trans_a ? input1_2 : input1_1;
    int k = trans_a ? input1_1 : input1_2; 
    int n = trans_b ? input2_1 : input2_2;
    int lda = m;
    int ldb = k;
    int ldc = m;

    cublas_wrapper->Gemm(
        transA,
        transB,
        m,
        n,
        k,
        input1->data,
        lda,
        input2->data,
        ldb,
        output->data,
        ldc,
        1.0f,
        0.0f
    );


    // 根据trans_c考虑是否需要返回转置的output
    if(trans_c){
        cublas_wrapper->Transpose(m,n,output->data);
    }
}



/************
 * 未完成
*/



// 仅考虑的是四维向量[a,b,c,d]*[a,b,d,c]
template<typename T>
void launchLinearStrideBatchGemm(   TensorWrapper<T>* input1, 
                                    TensorWrapper<T>* input2, 
                                    TensorWrapper<T>* output,
                                    cublasWrapper* cublas_wrapper,
                                    bool trans_a,
                                    bool trans_b,
                                    bool trans_c)
{
    cublasOperation_t transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    //目前支持n维矩阵无broadcast乘法
    LLM_CHECK_WITH_INFO(input1->shape.size() == input2->shape.size() 
                    &&  input1->shape.size() == output->shape.size(),
                    "Wrong shape in Batched Gemm, where broadcast is not available");
    int nrow = input1->shape.size() - 2;
    int ncol = input1->shape.size() - 1;

    // shape CHECK
    if(trans_a && trans_b){
        LLM_CHECK_WITH_INFO(input1->shape[nrow] == input2->shape[ncol],
                            "BatchGemm, A and B both trans, SOMETHING WORNG WITH SHAPE");
    }else if(trans_a && !trans_b){
        LLM_CHECK_WITH_INFO(input1->shape[nrow] == input2->shape[nrow],
                            "BatchGemm, A trans, SOMETHING WORNG WITH SHAPE");
    }else if(!trans_a && trans_b)
    {
        LLM_CHECK_WITH_INFO(input1->shape[ncol] == input2->shape[ncol],
                            "BatchGemm, B trans, SOMETHING WORNG WITH SHAPE");
    }else{
        LLM_CHECK_WITH_INFO(input1->shape[ncol] == input2->shape[nrow],
                            "BatchGemm, A and B no trans, SOMETHING WORNG WITH SHAPE");
    }
    
    int batchCount = 1;
    int count = 0;
    while(count < input1->shape.size()-2){
        LLM_CHECK_WITH_INFO(
            input1->shape[count] == input2->shape[count]
        &&  input1->shape[count] == output->shape[count],
        "Broad cast is not available, something wrong with batch_size");
        batchCount*= input1->shape[count];
        count++;
    }

    int m = !trans_a ? input1->shape[nrow] : input1->shape[ncol];
    int k = !trans_a ? input1->shape[ncol] : input1->shape[nrow];
    int n = !trans_b ? input2->shape[ncol] : input2->shape[nrow];
    int lda = m;
    int ldb = n;
    int ldc = m;
    int64_t strideA = m*k;
    int64_t strideB = k*n;
    int64_t strideC = m*n;

    
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