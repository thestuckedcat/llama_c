#pragma once
#include<cuda_runtime.h>
#include<cuda.h>
#include<cuda_fp16.h>
#include"src/utils/tensor.h"
#include"src/weights/llama_weight/norm_weights.h"
#include"src/utils/vectorize_utils.h"

template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out, //[num tokens, hidden_units]
                    TensorWrapper<T>* decoder_residual,   
                    LayerNormWeight<T>* attn_norm_weight,//RMSNorm weights
                    float eps//RMSnorm eps
                    );