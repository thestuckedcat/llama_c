#pragma once

//CUDA components
# include<cuda_runtime.h>
# include<cuda.h>
# include<cuda_fp16.h>

// self-defined tensor structure
# include "src/utils/tensor.h"

// EmbeddingWeight data structure
# include "src/weights/llama_weight/embedding_weights.h"


template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,
                          TensorWrapper<T>* output, 
                          EmbeddingWeight<T>* embed_table);

