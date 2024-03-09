/*
    * Maintenaned by StuckedCat
    * email: 529853411@qq.com

    * Contains following parts
    Index   Type            Name                    Description

    1.      CPU Function    cpuEmbedding            使用CPU完成Embedding

    2.      CPU Function    checkresults            检查cpu与gpu结果是否一致

    3.      main function                           (i)     参考llama确定了test基本参数并用随机数生成初始数据
                                                    (ii)    计算CPU embedding
                                                    (iii)   计算GPU embedding
                                                    (vi)    比较了结果

*/
# include <algorithm>           // std::fill_n
# include <math.h>              // expf, log, fabs(绝对值)
# include <stdlib.h>            // rand
# include <iostream>            // snprintf
# include <string>              // std::string
# include <vector>              // std::vector
# include <random>              

# include <cuda.h>
# include <cuda_fp16.h>
# include <cuda_runtime.h>

# include "src/utils/macro.h"   // #define CHECK
# include "src/kernels/input_embedding.h"   //embedding kernel


// This function calculate the embedding through CPU
void cpuEmbedding(  const int* input_ids, 
                    float* output, 
                    float* embed_table,
                    const int max_context_token_num,
                    const int hidden_size)
{
    for(int i = 0; i < max_context_token_num;i++){
        for(int j = 0; j < hidden_size;j++){
            output[j+i*hidden_size] = embed_table[j+input_ids[i] * hidden_size];
        }
    }
}


// This function get the cpu result ptr and gpu result ptr, 
// compare and output the abnormal value and its context
// float* cpu_output: malloc from cpu malloc
// float* gpu_output: malloc from cuda malloc
void checkResults(float* cpu_output, float* gpu_output, const int output_size){

    bool flag = true;

    // Data transmission, from gpu to cpu, first apply cpu memory to recieve data
    float* gpu_output_cpu = (float*) malloc(output_size * sizeof(float));
    //float* gpu_output_cpu = new float[output_size];

    //cudaMemcpy(destination ptr, src ptr, size in byte, direction)
    CHECK(cudaMemcpy(gpu_output_cpu, gpu_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    //compare
    for(int i = 0; i < output_size; i++){
        if(fabs(gpu_output_cpu[i] - cpu_output[i] > 1e-5)){
            //打印不匹配元素前后一共20个元素的值
            std::cout << "Dev : ";
            for(int j = max(0,i-10);j < min(output_size,i+10);j++){
                std::cout << gpu_output_cpu[i] << " ";
            }

            std::cout << std::endl;


            std::cout << "CPU : ";
            for(int j = max(0,i-10);j < min(output_size,i+10);j++){
                std::cout << cpu_output[i] << " ";
            }

            std::cout << std::endl;
        }
    }

    free(gpu_output_cpu);
    if(flag){
        std::cout << "CPU result is as same as GPU result" << std::endl;
    }else{
        std::cout << "CPU result is not same as GPU result" << std::endl;
    }
}


//  argv[1] -fp32: test fp32 GPU kernel
//          -fp16: test fp16 GPU kernel #TODO
//          -int8: test int8 GPU kernel #TODO
int main(int argc, char* argv[]){
    // llama参数大小
    const int max_context_token_num = 64;                               // 输入的词token个数
    const int hidden_size           = 4096;                             // 词嵌入维度
    const int vocab_size            = 30000;                            // 词汇库大小
    const int input_size            = max_context_token_num;            // 包含tokenID(int)的一个数组
    const int table_size            = vocab_size * hidden_size;         // vocab表[token_num]
    const int output_size           = max_context_token_num*hidden_size;// 输出input_token * hidden_size


    int* cpu_input = (int*) malloc(input_size * sizeof(int));
    float* cpu_table    = (float*)malloc(table_size * sizeof(float));   //用于存储vocabulary




    std::cout << "Initializing memory on host" << std::endl;


    // 配置随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    // 生成一个[0,vocab_size-1]的均匀分布的整数
    std::uniform_int_distribution<> dis_int(0,vocab_size - 1);
    // 生成一个[1.0,2.0]的实数均匀分布
    std::uniform_real_distribution<> dis_real(1.0,2.0);
    //使用:int random_int = dis_int(gen)
    //double random_double = dis_real(gen)

    //generate data
    for(int i = 0; i < max_context_token_num;i++){
        cpu_input[i] = dis_int(gen);
    }
    for(int i = 0; i < table_size;i++){
        //设置token的embedding,第1个token就是[1/hidden_size, 2/hidden_size,... ,1]
        cpu_table[i] = (float)(i / hidden_size);
    }





    // 计算CPU Embedding
    float* cpu_output = (float*)malloc(output_size * sizeof(float));
    cpuEmbedding(cpu_input, cpu_output, cpu_table, max_context_token_num, hidden_size);





    // 计算GPU Embedding

    // GPU参数准备
    int* gpu_input;
    float* gpu_table, *gpu_output;

    cudaMalloc((void**)&gpu_input, input_size * sizeof(int));
    cudaMalloc((void**)&gpu_table, table_size * sizeof(float));
    cudaMalloc((void**)&gpu_output, output_size * sizeof(float));

    std::cout << "init memory on device" << std::endl;
    CHECK(cudaMemcpy(gpu_input, cpu_input, input_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_table, cpu_table, table_size * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "Finish copy to device" << std::endl;


    //使用embedding launcher使用GPU计算
    DataType type_float = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    //TensorWrapper(device,datatype, vector:shape, data)
    TensorWrapper<int>* input_ids = new TensorWrapper<int>( Device::GPU, 
                                                            type_int,
                                                            {max_context_token_num, hidden_size},
                                                            gpu_input);
    TensorWrapper<float>* output = new TensorWrapper<float>(Device::GPU,
                                                            type_float,
                                                            {max_context_token_num, hidden_size},
                                                            gpu_output);
    EmbeddingWeight<float> emb_table;
    emb_table.data = gpu_table;


    // 启动内核，结果存在cudamalloc得到的output中
    launchInputEmbedding(input_ids, output, &emb_table);

    //对比结果-(malloc, cudamalloc, outputsize)
    checkResults(cpu_output, gpu_output, output_size);

    //释放内存
    cudaFree(gpu_output);
    cudaFree(gpu_table);
    cudaFree(gpu_input);

    //
    free(cpu_input);
    free(cpu_output);
    free(cpu_table);


}
