#include<stdio.h>
#include"src/kernels/rmsnorm.h"
#include"src/utils/tensor.h"
#include<iostream>
#include<fstream>
#include<random>
#include<cmath> //for sqrt
__host__ void CPU_RMSNorm_residual( float* decoder_out,
                                    float* scale, 
                                    float eps,
                                    int hidden_units,
                                    int num_tokens)
{
    for(int token_id = 0; token_id < num_tokens;token_id++){
        float denominator = 0.0f;
        float mean = 0.0f;
        float sum = 0.0f;
        // 计算1/sqrt(sum_x + eps)
        for(int embed_id = 0; embed_id < hidden_units;embed_id++){
            float temp = decoder_out[token_id * hidden_units + embed_id];
            sum += temp * temp;
        }

        std::cout << "CPU: The " << token_id << " th token sum is " << sum << std::endl;

        mean = (float)sum / hidden_units;

        denominator = 1.0f / std::sqrt(mean + eps);

        std::cout << "CPU: The " << token_id << " th token rsqrt is " << denominator << std::endl;
        //修改输出
        for(int embed_id = 0; embed_id < hidden_units; embed_id++){
            decoder_out[token_id * hidden_units + embed_id] *= denominator * scale[embed_id];
        }
    }
}


template<typename T>
bool CheckResult(   float* CPUoutput, 
                    T* GPUoutput, 
                    int hidden_units, 
                    int num_tokens)
{
    for(int i = 0;i < hidden_units * num_tokens;i++){
        if(fabs(CPUoutput[i] - (float)GPUoutput[i]) > 1e-5){
            printf("the %d th result is wrong, CPU = %f, GPU = %f\n",i, CPUoutput[i], (float)GPUoutput[i]);

            return false;
        }
    }
    return true;
}


template<typename T>
void GPU_RMSNorm_residual(  float* decoder_out,
                            float* scale,
                            float eps,
                            int hidden_units,
                            int num_tokens
                            )
{
    // 申请空间
    int total_size = hidden_units * num_tokens;

    // d_decoder_out
    float* d_decoder_out;
    CHECK(cudaMalloc((void**)&d_decoder_out, sizeof(float) * total_size));
    CHECK(cudaMemcpy(d_decoder_out, decoder_out, sizeof(float) * total_size, cudaMemcpyHostToDevice));

    // rsd
    float* d_decoder_rsd;
    CHECK(cudaMalloc((void**)&d_decoder_rsd, sizeof(float) * total_size));

    //scale
    float* d_scale;
    CHECK(cudaMalloc((void**)&d_scale, sizeof(float) * hidden_units));
    CHECK(cudaMemcpy(d_scale, scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));


    // TensorWrapper
    DataType type_float = getTensorType<float>();
    TensorWrapper<float>* decoder_out_tensor = new TensorWrapper<float>(    Device::GPU,
                            type_float,
                            {num_tokens,hidden_units},
                            d_decoder_out);

    TensorWrapper<float>* decoder_rsd = new TensorWrapper<float>(       Device::GPU,
            type_float,
            {num_tokens,hidden_units},
            d_decoder_rsd);

    LayerNormWeight<float> d_weight;
    d_weight.gamma = d_scale;


    // Active kernel
    launchRMSNorm<float>(decoder_out_tensor, decoder_rsd, &d_weight, eps);

    //Transfer Back to host
    CHECK(cudaMemcpy(decoder_out,decoder_out_tensor->data, sizeof(float)*total_size, cudaMemcpyDeviceToHost));

    cudaFree(d_decoder_out);
    cudaFree(d_decoder_rsd);
    cudaFree(d_scale);

}

int main(){
    const int num_tokens = 4;
    const int hidden_units = 4096;//使用float4不能超过4096，使用half2不能超过2048，目前不支持
    const int total_size = num_tokens * hidden_units;
    float eps = 1e-5;

    float* h_decoder_out,*d_decoder_out;
    h_decoder_out = (float*)malloc(sizeof(float) * total_size);
    d_decoder_out = (float*)malloc(sizeof(float) * total_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-2,2);
    for(int i = 0; i < total_size; i++){
        float temp = dis(gen);
        //float temp = (float)(i%2);
        h_decoder_out[i] = temp;
        d_decoder_out[i] = temp;
    }

    float* scale = (float*)malloc(sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units;i++){
        scale[i] = dis(gen);
        //scale[i] = 0.5f;
    }



    CPU_RMSNorm_residual(h_decoder_out, scale, eps, hidden_units,num_tokens);

    GPU_RMSNorm_residual<float>(d_decoder_out, scale, eps, hidden_units,num_tokens);


   bool result =  CheckResult<float>(h_decoder_out,d_decoder_out,hidden_units,num_tokens);

   if(result){
        std::cout << "GPU result is as same as CPU result" <<std::endl;
   }else{
        std::cout << "GPU result is NOT as same as CPU result" <<std::endl;
   }
}