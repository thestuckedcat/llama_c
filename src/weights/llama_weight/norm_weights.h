#pragma once

/*
 * RMSNorm(x) = gamma * (x_i) / sqrt(1/H * sum(x^2_i) + epsilon) 
 * 
 * 其中gamma是一个可学细的缩放参数
 * epsilon是一个防止除以0的极小常数
*/

template<typename T>
struct LayerNormWeight{
    T* gamma;
};