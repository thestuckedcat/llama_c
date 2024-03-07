/*
    * Maintenaned by StuckedCat
    * mail: 529853411@qq.com
    *
    * This file contains a embedding kernel and its launcher
    
    * Contains following parts
    Index   Type            Name                    Description

    1.      CUDA Kernel     embeddingFcuntor        利用CUDA为输入ID快速寻找词嵌入向量

    2.      CPU Function    launchInputEmbedding    根据llama2的权重shape启动1中的CUDA kernel

    3.      显式模板实例化   -                       - 

*/
# include<stdio.h>
# include "src/kernels/input_embedding.h"
# include "src/utils/cuda_debug_utils.cuh"
template<typename T>
__global__ void embeddingFunctor(   const int* input_tokenId,
                                    T* output,
                                    const T* embed_table,
                                    const int max_context_token_num,
                                    const int hidden_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    while(index < max_context_token_num * hidden_size){
        // tokenId handled by this thread
        int tokenId = input_tokenId[index/hidden_size];
        output[index] = embed_table[tokenId * hidden_size + index % hidden_size];
        index += blockDim.x * gridDim.x;
    } 
}


template<typename T>
void launchInputEmbedding(  TensorWrapper<int>* input_ids,      //输入INT[token num]
                            TensorWrapper<T>*   output,         //输出[token num, hidden size]
                            EmbeddingWeight<T>* embed_table     //原始构建的vocabulary词表,[vocal_size, vec_size]
    ){
        const int blockSize = 256;
        const int max_context_token_num = output->shape[0];
        const int hidden_size = output->shape[1];
        const int gridSize = 2048;
        assert(max_context_token_num == inpout_ids->shape[0]);//这两个需要相同

        embeddingFunctor<T><<<gridSize, blockSize>>>(   input_ids->data,
                                                        output->data, 
                                                        embed_table->data,
                                                        max_context_token_num,
                                                        hidden_size);
    }



//显式模板实例化
//为什么不放到.h中？这是因为如果以后使用.cpp引用.h文件，那么一些CUDA独有的符号，例如<<<>>>就会被引入.cpp，而Cpp并没有对应的声明，因此会报错，
//另外可以控制模板实例化的位置，同时避免了编译器在每个包含模板定义的文件中都生成模板实例的冗余。
// 效果与重载很相似
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<float>* output,       
                                   EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<half>* output,       
                                   EmbeddingWeight<half>* embed_table);
