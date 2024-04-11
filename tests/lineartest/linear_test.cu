#include "src/kernels/linear.h"

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


