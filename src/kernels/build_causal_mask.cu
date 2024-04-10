#include "src/kernels/build_causal_mask.h"

/*
    输出:mask:
    输入:qlength: [batch_size]也就是句子个数
        klength: [batch_size]
        mask: [batch_size, max_q_length,max_k_length]

        max_k_length:当前轮次最大的上下文长度,max(k_lens)
        max_q_length:当前轮次batch中最大的句子长度, max(q_lens)

        输入的klength就是context length，qlength就是decoder输入的length

    此处mask是考虑过去上下文的，也就是mask考虑过去上下文的长度，

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