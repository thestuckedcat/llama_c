#include<algorithm>
#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<string>
#include<vector>

#include "src/kernels/paddingoffset.h"
/*
    生成了输入Input：一个batch内每个句子的长度
    获得输出：
    paddingoffset:每个句子的token在padding后应该向前移动以回归padding前的距离
    cumsum: 一个batch内句子长度的前缀和。
*/
int main(){
    const int batch_size = 5;
    const int max_q_len = 10;

    std::cout << "Begin padding offset Test" << std::endl;
    std::cout << "set batch_size to" << batch_size << std::endl;
    std::cout << "set text max length to" << max_q_len << std::endl;
    DataType type_int = getTensorType<int>();

    //每个句子的长度集合
    int *cpu_sequence;
    int *gpu_sequence;
    cpu_sequence = (int*)malloc(sizeof(int)*batch_size);
    CHECK(cudaMalloc((void**)&gpu_sequence, sizeof(int)*batch_size));
    std::cout << "The length of each sequence in Batch is "<< std::endl;
    
    for(int i = 0; i < batch_size; i++){
        cpu_sequence[i] = batch_size;
        std::cout << cpu_sequence[i] << " ";
    }
    std::cout << std::endl;
    CHECK(cudaMemcpy(gpu_sequence, cpu_sequence, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    TensorWrapper<int>* input = new TensorWrapper<int>( Device::GPU, 
                                                        type_int, 
                                                        {batch_size},
                                                        gpu_sequence);

    // output:前缀和
    int* cpu_cum_sum;
    int* gpu_cum_sum;
    cpu_cum_sum = (int*) malloc(sizeof(int) * (batch_size+1));
    CHECK(cudaMalloc((void**)&gpu_cum_sum, sizeof(int) * (batch_size + 1)));
    TensorWrapper<int>* cum_sum = new TensorWrapper<int>(   Device::GPU,
                                                            type_int,
                                                            {batch_size+1},
                                                            gpu_cum_sum);

    // output: padding表
    int* cpu_padding_offset;
    int* gpu_padding_offset;
    cpu_padding_offset = (int*)malloc(sizeof(int)*batch_size*max_q_len);
    CHECK(cudaMalloc((void**)&gpu_padding_offset, sizeof(int)*batch_size*max_q_len));
    TensorWrapper<int>* padding_offset = new TensorWrapper<int>(Device::GPU, 
                                                                type_int, 
                                                                {batch_size, max_q_len},
                                                                gpu_padding_offset);



    launchCalpaddingoffset( padding_offset,
                            cum_sum,
                            input,
                            Device::GPU);

    
    CHECK(cudaMemcpy(cpu_cum_sum, gpu_cum_sum, sizeof(int)*(batch_size+1), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_padding_offset, gpu_padding_offset, sizeof(int)*(batch_size * max_q_len), cudaMemcpyDeviceToHost));



    for(int i = 0; i < batch_size*max_q_len;i++){
        printf("padding_offset = %d\n", cpu_padding_offset[i]);
    }

    for(int i = 0; i < batch_size+1;i++){
        printf("cum_sum=%d\n", cpu_cum_sum[i]);
    }


    free(cpu_cum_sum);
    free(cpu_padding_offset);
    free(cpu_sequence);


    cudaFree(gpu_cum_sum);
    cudaFree(gpu_sequence);
    cudaFree(gpu_padding_offset);

}