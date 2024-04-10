#include "src/kernels/build_causal_mask.h"
#include "src/utils/tensor.h"
#include "src/utils/macro.h"
#include<random>
/*
    参数:
    batch_size就是句子个数，max_k_len就是上下文带回答的长度，max_q_len就是回答的长度

    mask: [batch_size, max_q_len, max_k_len]的01矩阵，其中1代表可访问/在视野内

    q_lens: [batch_size]， 


*/



void CPU_casualMask(float* mask,
                    const int* q_lens,
                    const int* k_lens,
                    int max_q_len,
                    int max_k_len,
                    int batch_size)
{
    for(int batch_id = 0;batch_id < batch_size;batch_id++){
        // 当前句子开始的地方
        int start = batch_id * max_q_len * max_k_len;

        int q = q_lens[batch_id];
        int k = k_lens[batch_id];

        for(int query_id = 0; query_id < max_q_len;query_id++){
            for(int key_id = 0; key_id < max_k_len;key_id++){
                if(key_id <= query_id+(k - q) && query_id < q && key_id < k){
                    mask[start + query_id * max_k_len + key_id] = 1.0f;
                }else{
                    mask[start + query_id * max_k_len + key_id] = 0.0f;
                }
            }
        }

    }
    std::cout <<"CPU calculation complete" << std::endl;
}       

void GPU_casualMask(float* mask,
                    const int* q_lens,
                    const int* k_lens,
                    int max_q_len,
                    int max_k_len,
                    int batch_size)
{
    float* d_mask;
    CHECK(cudaMalloc((void**)&d_mask, sizeof(float) * max_q_len * max_k_len * batch_size));

    int* d_q_lens;
    int* d_k_lens;
    CHECK(cudaMalloc((void**)&d_q_lens, sizeof(int) * batch_size));
    CHECK(cudaMalloc((void**)&d_k_lens, sizeof(int) * batch_size));
    CHECK(cudaMemcpy(d_q_lens,q_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k_lens,k_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice));

    DataType type_float = getTensorType<float>();
    TensorWrapper<float>* mask_tensor = new TensorWrapper<float>(   Device::GPU,
                                                                    type_float,
                                                                    {batch_size,max_q_len,max_k_len},
                                                                    d_mask);

    DataType type_int = getTensorType<int>();
    TensorWrapper<int>* q_lens_tensor = new TensorWrapper<int>( Device::GPU,
                                                                type_int,
                                                                {batch_size},
                                                                d_q_lens);

    TensorWrapper<int>* k_lens_tensor = new TensorWrapper<int>( Device::GPU,
                                                                type_int,
                                                                {batch_size},
                                                                d_k_lens);                                                                                                               

    launchBuildCausalMasks(mask_tensor, q_lens_tensor,k_lens_tensor);






    CHECK(cudaMemcpy(mask, mask_tensor->data, sizeof(float) * batch_size * max_q_len * max_k_len, cudaMemcpyDeviceToHost));


    cudaFree(d_mask);
    cudaFree(d_q_lens);
    cudaFree(d_k_lens);
}



bool CheckResult(float* CPUres, float* GPUres, const int size) {
    for(int i = 0; i < size; i++) {
        if(fabs(CPUres[i] - GPUres[i]) > 1e-6){
            printf("the %dth res is wrong, CPU mask = %f, GPU mask = %f\n", i, CPUres[i], GPUres[i]);
            return false;
        }
    }
    return true;
}

void print_matrix(float* h_mask,float* d_mask, int batch_size, int max_k_len, int max_q_len){
    std::cout << "Left is on CPU and Right is on GPU" <<std::endl;
    for(int i = 0; i < batch_size;i++){
        std::cout << "batch " << i << std::endl; 
        for(int j = 0; j < max_q_len;j++){
            for(int k = 0;k < max_k_len;k++){
                std::cout << h_mask[j * max_q_len + k] << " ";
            }

            std::cout << "      " ;
            for(int k = 0;k < max_k_len;k++){
                std::cout << d_mask[j * max_q_len + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

int main(){
    std::cout << "Test for casual_mask begin"  << std::endl;
    const int batch_size = 2;
    const int max_q_len = 10;
    const int max_k_len = 10;

    const int mask_size = batch_size * max_q_len * max_k_len;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1,max_q_len);

    float* d_mask, *h_mask;
    int* q_lens, *k_lens;

    d_mask = (float*)malloc(sizeof(float) * mask_size);
    h_mask = (float*)malloc(sizeof(float) * mask_size);

    q_lens = (int*) malloc(sizeof(int) * batch_size);
    k_lens = (int*) malloc(sizeof(int) * batch_size);

    for(int i = 0; i < batch_size; i++){
        q_lens[i] = dis(gen);
        k_lens[i] = q_lens[i];
    }

    CPU_casualMask(h_mask, q_lens, k_lens, max_q_len, max_k_len, batch_size);
    GPU_casualMask(d_mask, q_lens, k_lens, max_q_len, max_k_len, batch_size);

    bool res = CheckResult(h_mask, d_mask, mask_size);

    if(res){
        std::cout << "All data true" << std::endl;
    }else{
        std::cout << "some data not match" << std::endl;
    }

    print_matrix(h_mask, d_mask, batch_size, max_k_len,max_q_len);
    free(d_mask);
    free(h_mask);
    free(q_lens);
    free(k_lens);
}



