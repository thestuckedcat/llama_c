# pragma once

# include<vector>
# include<cstdint>
# include<cuda_fp16.h>


// 强类型枚举，这代表我们即将使用的权重参数的可能类型
enum class WeightType{
  FP32_W,
  FP16_W,
  INT8_W,
  UNSUPPORTED_W
};

//用于判定某个数据类型T的类型是什么

template<typename T>
inline WeightType getWeightType(){
  if(std::is_same<T,float>::value || std::is_same<T,const float>::value){
    return WeightType::FP32_W;
  }
  else if(std::is_same<T,half>::value || std::is_same<T,const half>::value){
    return WeightType::FP16_W;
  }
  else if(std::is_same<T,int8_t>::value || std::is_same<T,const int8_t>::value){
    return WeightType::INT8_W;
  }
  else{
    return WeightType::UNSUPPORTED_W;
  }
}


template<typename T>
class BaseWeight{
public:
  // Weight类型
  WeightType wtype;
  // Weight tensor的形状
  std::vector<int> shape;
  // Weight本身也具有Bias，将Bias也归为Weight类，当为Bias时data为nullptr
  T* data;

  T* bias;

};