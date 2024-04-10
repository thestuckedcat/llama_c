#include "src/kernels/build_causal_mask.h"

/*
    参数:
    batch_size就是句子个数，

    max_q_len：所有batch中的最大query个数

    max_k_len: 所有batch中的最大key个数

    mask: [batch_size, max_q_len, max_k_len]的01矩阵，其中1代表可访问/在视野内

    q_lens: [batch_size]， 每个句子的query

    k_lens: [batch_size], 每个句子的key


    在本次实现的Decoder-only架构中，每个句子会通过self-masked-attention，因此q_lens与k_lens应该是相同的。

    这种写法是常规的Transformer写法，考虑了Decoder的attention使用Encoder的KV以及自身的Q，此时会导致q_lens和k_lens不同的情况。


*/

template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(   T* mask,
                                                            const int* q_lens,
                                                            const int* k_lens,
                                                            int max_q_len,
                                                            int max_k_len){
    
    //处理max_q_len*max_k_len大小的mask矩阵                                                                
    int qlen = q_lens[blockIdx.x];//行
    int klen = k_lens[blockIdx.x];//列
    // 挑选出每个block需要处理的batch
    mask+= blockIdx.x * max_q_len * max_k_len;

    int offset = threadIdx.x;
    while(offset < max_q_len * max_k_len){
        int q = offset / max_k_len;
        int k = offset % max_k_len;

        bool is_one = q < qlen && k < klen && k<= q + (klen - qlen) && k >= klen-qlen;
        mask[offset] = static_cast<T>(is_one);

        offset += blockDim.x;
    }

}


template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask,
                            TensorWrapper<int>* q_lens,
                            TensorWrapper<int>* k_lens)
{
    // 每个block处理一个句子，因此是batch_size个block
    // 每个block使用256个线程处理[max_q_length,max_k_length]的空间
    int batch_size = mask->shape[0];
    int max_q_len = mask->shape[1];
    int max_k_len = mask->shape[2];
    BuildCausalMasksConsideringContextPastKV<T><<<batch_size, 256>>>(mask->data,q_lens->data, k_lens->data,max_q_len,max_k_len);
}

template void launchBuildCausalMasks(   TensorWrapper<float>* mask,
                                        TensorWrapper<int>* q_lens,
                                        TensorWrapper<int>* k_lens);

template void launchBuildCausalMasks(   TensorWrapper<half>* mask,
                                        TensorWrapper<int>* q_lens,
                                        TensorWrapper<int>* k_lens);