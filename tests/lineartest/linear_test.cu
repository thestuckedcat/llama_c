#include "src/kernels/linear.h"
#include<vector>
#include<iostream>
#include<random>
#include"src/kernels/cublas_utils.h"
#include"utils/tensor.h"
/*
    测试两种矩阵乘法

    第一种:普通矩阵乘

            Input shape = [sequence len, hidden_size]
            weight shape = [hidden_size, hidden_size]


    第二种：softmax之前的linear

            Input shape = [batch_size, hidde]
            weight shape = [vocabulary_size, hidden_size]
            trans_b = true


    测试两种批量矩阵乘

    第一种：

            q shape: [batch_size, head_nums, sequence_len, head size]
            k shape: [batch_size, head_nums, sequence_len, head size]
            trans_b = true

    第二种：

            qk shape: [batch_size, head_nums, sequence_len, sequence_len]
            v shape : [batch_size, head_nums, sequence_len, head_size]


    默认 A = [m,k], B = [k,n], C = [m,n]


    同时，考虑如下情况：
    
    输入的是行主序的
    A * B = C
    考虑到我们使用的Cublas是列主序的，因此
    我们就需要通过is_row_leading
*/

// 以下传入的数据皆为行主序


void CPU_gemm(  const std::vector<std::vector<int>>& shape,
                std::vector<float*> &data,
                bool trans_a,
                bool trans_b)
{
    //data->[left_matrix, right_matrix, cpu_output_matrix, gpu_output_matrix]，输出矩阵都已归零
    //shape->input1, input2, output->[dim1,dim2,dim3,dim4]，在这里[dim3,dim4]才是用来计算的矩阵，其他是批次
    // 此处考虑batch的shape都一样
    int loc_row = shape[0].size()-2;
    int loc_col = shape[0].size()-1;

    int batch_size = 1;
    int count = 0;
    while(count < shape[0].size() - 2){
        batch_size *= shape[0][count++];
    }

    // m,k,n
    int m = !trans_a ? shape[0][loc_row] : shape[0][loc_col];
    int k = !trans_a ? shape[0][loc_col] : shape[0][loc_row];
    int n = !trans_b ? shape[1][loc_col] : shape[1][loc_row];

    int input1_matrix_size = m * k;
    int input2_matrix_size = k * n;
    int output_matrix_Size = m * n;

    for(int b_id = 0; b_id < batch_size; b_id++)
    {
        float* input1_matrix = data[0] + b_id * input1_matrix_size;
        float* input2_matrix = data[1] + b_id * input2_matrix_size;
        float* output_matrix = data[2] + b_id * output_matrix_Size;


        for(int i = 0; i < m;i++)
        {
                // left_matrix的每行
                for(int j = 0; j < n;j++)
                {
                    // right_matrix的每一列
                    for(int p = 0; p < k;p++)
                    {
                        // elements do calculation
                        // input1 [i, p] + input2 [p, j] = output [i, j]
                        if(!trans_a && !trans_b)
                            output_matrix[i * n + j] += input1_matrix[i * k + p] * input2_matrix[p * n + j];
                        else if(!trans_a && trans_b)
                            output_matrix[i * n + j] += input1_matrix[i * k + p] * input2_matrix[j * k + p];
                        else if(trans_a && !trans_b)
                            output_matrix[i * n + j] += input1_matrix[p * m + i] * input2_matrix[p * n + j];
                        else
                            output_matrix[i * n + j] += input1_matrix[p * m + i] * input2_matrix[j * k + p];

                        /*
                        int idxA = trans_a ? p * m + i : i * k + p;
                        int idxB = trans_b ? j * k + p : p * n + j;
                        // Accumulate the product into the output element
                        output_matrix[i * n + j] += input1_matrix[idxA] * input2_matrix[idxB];
                        */
                    }
                }
        }
    }

}



// 以下的环节必须考虑行主序输入时，使用转置变换为列主序的形式，并确定调用cudakernel时的左右矩阵顺序， 未完成
void GemmTest_Type1_rowleadingInput(const std::vector<std::vector<int>>& shape,
                                    std::vector<float*> &data,
                                    bool trans_a, 
                                    bool trans_b)
{

    // shape:input1, input2, h_output, d_output

    DataType float_type = getTensorType<float>();

    TensorWrapper<float> *input1 = new TensorWrapper(   Device::GPU, 
                                                        DataType::float_type,
                                                        shape,
                                                        data[0]);

    TensorWrapper<float> *input2 = new TensorWrapper(   Device::GPU,
                                                        DataType::float_type,
                                                        shape,
                                                        data[1]);

    TensorWrapper<flaot> *output = new TensorWrapper(   Device::GPU,
                                                        DataType::float_type,
                                                        shape,
                                                        data[3]);


}                                                        
void GemmTest_Type2_rowleadingInput(std::vector<float*> data,
                                    const std::vector<int>& shape,                     
                                    cublasWrapper* cublas_wrapper,
                                    bool is_row_leading)
{

}

void BatchedGemmTest_Type1( std::vector<float*> data, 
                            const std::vector<int>& shape,
                            cublasWrapper* cublas_wrapper,
                            bool is_row_leading)
{

}

void BatchedGemmTest_Type2( std::vector<float*> data,
                            const std::vector<int>& shape,
                            cublasWrapper* cublas_wrapper,
                            bool is_row_leading)
{

}

bool CheckResult(float* CPU_res, float* GPU_res, int nsize)
{

}


std::vector<float*> generate_data(const std::vector<std::vector<int>>& shape){
    // shape:input1 shape, input2 shape, output shape
    // 存入input1,input2,h_output, d_output
    std::vector<float*> res;

    int total_size_input1 = 1;
    int total_size_input2 = 1;
    int total_size_output = 1;

    for(int i = 0; i < shape[0].size();i++){
        total_size_input1 *= shape[0][i];
        total_size_input2 *= shape[1][i];
        total_size_output *= shape[2][i];
    }

    float* input1   =   (float*)malloc(sizeof(float) * total_size_input1);
    float* input2   =   (float*)malloc(sizeof(float) * total_size_input2);
    float* h_output =   (float*)malloc(sizeof(float) * total_size_output);
    float* d_output =   (float*)malloc(sizeof(float) * total_size_output);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-2,2);

    for(int i = 0; i < total_size_input1;i++){
        input1[i] = dis(gen);
    }
    
    for(int i = 0; i < total_size_input2;i++){
        input2[i] = dis(gen);
    }

    for(int i = 0; i < total_size_output; i++){
        d_output[i] = 0;
        h_output[i] = 0;
    }
    res.push_back(input1);
    res.push_back(input2);
    res.push_back(h_output);
    res.push_back(d_output);


    return res;
}