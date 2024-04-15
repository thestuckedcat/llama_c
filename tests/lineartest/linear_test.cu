#include "src/kernels/linear.h"
#include<vector>
#include<iostream>
#include<random>
#include"src/kernels/cublas_utils.h"
#include"utils/tensor.h"
#include<algorithm>
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



// 以下的环节必须考虑行主序输入时，使用转置变换为列主序的形式，并确定调用cudakernel时的左右矩阵顺序
void GPU_gemm(const std::vector<std::vector<int>>& shape,
                                    std::vector<float*> &data,
                                    bool is_row_leading,
                                    cublasWrapper* cublas_wrapper,
                                    bool trans1,
                                    bool trans2)
{

    // data:input1, input2, h_output, d_output
    // shape: input1, input2, output
    int input1_size = std::accumulate(shape[0].begin(),shape[0].end,1,std::multiplies<int>());
    int input2_size = std::accumulate(shape[1].begin(),shape[1].end,1,std::multiplies<int>());
    int output_size = std::accumulate(shape[3].begin(),shape[3].end,1,std::multiplies<int>());

    // Apply cuda resources
    float* input1;
    CHECK(cudaMalloc((void**)&input1, input1_size * sizeof(float)));
    CHECK(cudaMemcpy(input1, data[0], input1_size * sizeof(float), cudaMemcpyHostToDevice));
    float* input2;
    CHECK(cudaMalloc((void**)&input2, input2_size * sizeof(float)));
    CHECK(cudaMemcpy(input2, data[1], input2_size * sizeof(float), cudaMemcpyHostToDevice));
    float* output;
    CHECK(cudaMalloc((void**)&output, output_size * sizeof(float)));

    // Wrap
    DataType float_type = getTensorType<float>();

    TensorWrapper<float> *left_matrix;

    TensorWrapper<float> *right_matrix;

    TensorWrapper<float> *output_matrix = new TensorWrapper( Device::GPU,
                                                    float_type,
                                                    shape[3],
                                                    output);




    bool trans_a, trans_b, trans_c;
    if(is_row_leading){
        // 如果为行主序，需要转成列主序,C = A * B->C^T = B^T * A^T
        // 两步： 交换左右矩阵顺序，dimension判定是否trans
        // dimension依据以下顺序判断：是否已经满足矩阵乘要求，是否右矩阵需要转置，是否左矩阵需要转置，是否都需要转置
        left_matrix = new TensorWrapper(Device::GPU,
                                        float_type,
                                        shape[1],
                                        input2);
        right_matrix = new TensorWrapper(   Device::GPU,
                                            float_type,
                                            shape[0],
                                            input1);                                       

        // 因为交换了顺序，所以需要多进行一次trans
        trans_a = !trans2;
        trans_b = !trans1;
        trans_c = true;
        
    }else{
        // 列主序，已经满足要求
        left_matrix = new TensorWrapper(Device::GPU,
                                        float_type,
                                        shape[0],
                                        input1);
        
        right_matrix = new TensorWrapper(Device::GPU,
                                        float_type,
                                        shape[1],
                                        input2);

        // 保留trans
        trans_a = trans1;
        trans_b = trans2;
        trans_c = false;
    }

    // 判断shape是否正确
    int dim_size = left_matrxi.shape.size();
    int row_a = !trans_a ? left_matrix.shape[dim_size - 2] : left_matrix.shape[dim_size - 1];
    int col_a = !trans_a ? left_matrix.shape[dim_size - 1] : left_matrix.shape[dim_size - 2];
    int row_b = !trans_b ? right_matrix.shape[dim_size - 2] : right_matrix.shape[dim_size - 1];
    int col_b = !trans_b ? right_matrix.shape[dim_size - 1] : right_matrix.shape[dim_size - 2];

    if(col_a == row_b){
        //满足要求
    }else if(row_a == row_b){
        !trans_a;
    }else if(col_a == col_b){
        !trans_b;
    }else if(row_a == col_b){
        !trans_a;
        !trans_b;
    }

    if(shape[0].size() == 2)
        launchLinearGemm(   left_matrix,
                            right_matrix, 
                            output_matrix,
                            cublas_wrapper,
                            trans_a,
                            trans_b,
                            trans_c);
    else if(shape[0].size() > 2)
        launchLinearStrideBatchGemm(left_matrix,
                                    right_matrix,
                                    output_matrix,
                                    cublas_wrapper,
                                    trans_a,
                                    trans_b,
                                    trans_c);

    CHECK(cudaMemcpy(data[3], output->data, output_size * sizeof(float), MemcpyDeviceToHost));

    cudaFree(input1);
    cudaFree(input2);
    cudaFree(output);
    delete left_matrix;
    delete right_matrix;
    delete output_matrix;
}                                                        

bool CheckResult(const std::vector<std::vector<int>>& shape,
                                    std::vector<float*> &data)
{
    //data->[left_matrix, right_matrix, cpu_output_matrix, gpu_output_matrix]，输出矩阵都已归零
    //shape->input1, input2, output->[dim1,dim2,dim3,dim4]，在这里[dim3,dim4]才是用来计算的矩阵，其他是批次

    int output_size = std::accumulate(shape[3].begin(),shape[3].end(), 1, std::multiplies<int>());

    for(int i = 0; i < output_size; i++){
        if(fabs(data[2][i] - data[3][i]) > 1e-5){
            std::cout << "Wrong data " << std::endl;
            std::cout << "cpu data:  " << data[2][i] << std::endl;
            std::cout << "gpu data:  " << data[3][i] << std::endl;
            return false;
        }
    }
    std::cout << "All data remain true" << std::endl;
    return true;

}


std::vector<float*> generate_data(const std::vector<std::vector<int>>& shape){
    // shape:input1 shape, input2 shape, output shape
    // 存入input1,input2,h_output, d_output
    std::vector<float*> res;

    int total_size_input1 = std::accumulate(shape[0].begin(), shape[0].end(), 1, std::multiplies<int>());
    int total_size_input2 = std::accumulate(shape[1].begin(), shape[1].end(), 1, std::multiplies<int>());
    int total_size_output = std::accumulate(shape[2].begin(), shape[2].end(), 1, std::multiplies<int>());

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


int main(){
    std::vector<std::vector<int>> shape{{}}
    std::vector<float>* data = generate_data(shape, trans_a, trans_b);

    CPU_gemm(shape, data, trans_a, trans_b);
    GPU_gemm(shape, data, is_row_leading, cublas_wrapper, trans_a, trans_b);
}