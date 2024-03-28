#pramga once
#include<cuda_runtime.h>
#include<cuda.h>
#include<cuda_fp16.h>
#include"src/utils/tensor.h>"
#include"src/weights/llama_weight/norm_weights.h"
#include"src/utile/vectorize_utils.h"

template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_in,// [num tokens, hidden units]
                    LayerNormWeight<T>& attention_norm_weight, // gamma
                    float epsilon);
