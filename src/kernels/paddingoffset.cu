# include "src/kernels/paddingoffset.h"

__global__ void paddingoffset_kernel(   int* padding_offset,
                                        int* cum_sum,
                                        const int* input_lengths,
                                        const int batch_size, 
                                        const int largest_length
                                        )
{
    int total_seqlen = 0;//维护前缀和
    int offset = 0;//维护偏移的前缀和
    int idx = 0;
    for(int b = 0; b < batch_size;b++){
        int seqlen = input_lengths[b];
        cum_sum[b] = total_seqlen;

        for(int i = 0; i < seqlen;i++){
            padding_offset[idx++] = offset;
        }
        offset += largest_length - seqlen;
        total_seqlen += seqlen;
    }
    cum_sum[batch_size] = total_seqlen;
}


void launchCalpaddingoffset(TensorWrapper<int>* paddingoffset, 
                            TensorWrapper<int>* cum_seqlens,
                            TensorWrapper<int>* input_lengths,
                            Device device)
{
    if(device == Device::GPU)
    {
        const int batch_size = paddingoffset->shape[0];
        const int largest_length = paddingoffset->shape[1];

        LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0],
                            "input_lengths number should be the same as batch_size");
        LLM_CHECK_WITH_INFO(batch_size + 1 == cum_seqlens->shape[0],
                            "cum sequence length numbers should be equal to batchsize + 1");

        paddingoffset_kernel<<<1,1>>>(paddingoffset->data, cum_seqlens->data, input_lengths->data, batch_size, largest_length
    );
    }
    else if(device == Device::CPU){
        const int batch_size = paddingoffset->shape[0];
        const int largest_length = paddingoffset->shape[1];

        LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0],
                            "input_lengths number should be the same as batch_size");
        LLM_CHECK_WITH_INFO(batch_size + 1 == cum_seqlens->shape[0],
                            "cum sequence length numbers should be equal to batchsize + 1");


        int total_seqlen = 0;//维护前缀和
        int offset = 0;//维护偏移的前缀和
        int idx = 0;
        for(int b = 0; b < batch_size;b++){
            int seqlen = input_lengths->data[b];
            cum_seqlens->data[b] = total_seqlen;

            for(int i = 0; i < seqlen;i++){
                paddingoffset->data[idx++] = offset;
            }
            offset += largest_length - seqlen;
            total_seqlen += seqlen;
        }
        cum_seqlens->data[batch_size] = total_seqlen;
    }

        

}