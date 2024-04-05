#include<stdio.h>
#include"src/kernels/rmsnorm_kernel.h"

// 函数接受每个线程的val，函数返回warp内所有这些值的总和(thread0)。
template<typename T>
__device__ T warpReduceSum(T val){
    unsigned int active_mask = __activemask();
    for(int i = 32/2; i > 0; i >>=1){
        val += __shfl_xor_sync(active_mask, val, i); // Synchronize all threads in warp and get "value" from lane i
    }
    return val;// 注意，每个线程都会执行，但是只有thread0的值才是我们需要的
}


template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warpid = tid/32;
    int laneid = tid%32;

    int warpnum = (blockDim.x + 31) / 32;
    val = warpReduceSum<T>(val);

    /*
    // 这种方法限制了并行性
    __shared__ T warpsum = 0;
    __syncthreads();

    if(laneid == 0){
        atomicAdd(&warpsum, val);
    }
    __syncthreads();
    
    return warpsum; 
    */

    __shared__ T warpsum[32];//考虑blocksize超过1024性能下降， 因此默认blocksize最大1024，设置最大32的数组

    // 存入warpsum
    if(laneid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();
    // 为block前warpnum个thread分配这些sum，然后使用warpreduce再次计算
    T sum = tid < warpnum ? warpsum[tid] : (T)0;
    sum = warpReduceSum<T>(sum);

    // 因为最大不超过32个，因此一个warpReduceSum就可以解决
    return sum;
}

// RMSNorm(x) = gamma * (x_i) / sqrt(1/H * sum(x^2_i) + epsilon) 
template<typename T>
__global__ void RMSNorm(T* decoder_in,  //[num tokens(batch size), q_hidden_units]
                        T* decoder_residual,//残差连接就相当于我们将残差输入另做备份到需要的层进行计算，nullptr意味着当前层不使用残差连接
                        T* gamma,       // [q_hidden_units], RMSNorm weights
                        float epsilon,  
                        int num_tokens, 
                        int hidden_units){
    // 平方每个数据
    int vec_size = Vec<T>::size();
    using Vec_t = typename Vec<T>::type;

    float thread_sum = 0.0f;
    // 将当前block对应的decoder_in数据切割
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_in + blockIdx.x * hidden_units);//注意我们的blockDim.x实际上是hidden_units/vec_size，这里容易搞混
    if(decoder_residual != nullptr)
        Vec_t* rsd;
        rsd = reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_units);
    for(int idx = threadIdx.x; idx < hidden_units/vec_size; idx+=blockDim.x){
        //每个线程取出一个float4
        Vec_t vec = dout[idx];
        if(decoder_residual != nullptr)
            rsd[idx] = vec;//保存当前值作为residual
        // 平方
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;
    }
    thread_sum = blockReduceSum<float>(thread_sum);





    // 因为均值是block层面，因此使用shared memory
    __shared__ float inv_mean;
    // 对于每个block的
    if(threadIdx.x == 0){
        //快速计算float倒数平方根(reciprocal square root)
        inv_mean = rsqrtf(thread_sum / hidden_units + epsilon);
    }
    __syncthreads();

    //修改输出
    Vec_t* g = reinterpret_cast<Vec_t*>(gamma);
    for(int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x){
        Vec_t vec = dout[idx];
        dout[idx].x = vec.x * inv_mean * g[idx].x;
        dout[idx].y = vec.y * inv_mean * g[idx].y;
        dout[idx].x = vec.z * inv_mean * g[idx].z;
        dout[idx].x = vec.w * inv_mean * g[idx].w;
        
    }
}


//half2版本
template <>
__global__ void RMSNorm(half* decoder_out, // [num tokens, q_hidden_units]
                        half* decoder_residual,
                        half* scale, //[q_hidden_units], RMSNorm weights
                        float eps, //RMSNorm eps
                        int num_tokens, 
                        int hidden_units){
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t* s; 
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);
    Vec_t* rsd;
    if (decoder_residual != nullptr) {
        rsd = reinterpret_cast<Vec_t*>(decoder_residual + batch_id * hidden_units);
    }
    float thread_accm = 0.0f;
    // 考虑到blockReduceSum使用float，首先转换为float来计算
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t out = dout[i];// note the offset should divide vec size
        if (decoder_residual != nullptr) {
            rsd[i] = out;
        }
        thread_accm += __half2float(out.x) * __half2float(out.x);
        thread_accm += __half2float(out.y) * __half2float(out.y);
    } //x^2
    
    // mean(x^2)
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){
        inv_fenmu = rsqrtf(float(blocksum / hidden_units) + eps);
    }
    __syncthreads();
    // rmsnorm，需要转换回half
    s = reinterpret_cast<Vec_t*>(scale);
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t dout_h2 =dout[i];
        dout[i].x = s[i].x * __float2half(__half2float(dout_h2.x) * inv_fenmu);
        dout[i].y = s[i].y * __float2half(__half2float(dout_h2.y) * inv_fenmu);
    }    
}


template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out, //[num tokens, hidden_units]
                    TensorWrapper<T>* decoder_residual,   
                    LayerNormWeight<T>& attn_norm_weight,//RMSNorm weights
                    float eps,//RMSnorm eps
                    bool is_last// print last rmsnorm output
                    ){
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;

    int num_threads = hidden_units / vec_size;

    T* rsd = decoder_residual->data;

    dim3 grid(num_tokens);
    dim3 block(num_threads);
    RMSNorm<T><<<grid,block>>>( decoder_out->data,
                                rsd,
                                attn_norm_weight,
                                eps,
                                num_tokens,
                                hidden_units);

    
}


template void launchRMSNorm( TensorWrapper<float>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<float>* decoder_residual,
                    LayerNormWeight<float>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last
                    );
template void launchRMSNorm( TensorWrapper<half>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<half>* decoder_residual,
                    LayerNormWeight<half>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last
                    );




