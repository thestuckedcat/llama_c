/*
  * Maintenaned by StuckedCat
  * mail: 529853411@qq.com

  This headfile define a general used data structure Tensor

  contains following parts
  Index:    Type:                 Name:                   Description:

  1.        enumeration:          Device                  Enumerate Possible devices

  2.        enumeration:          DataType                Enumerate Possible datatype(not only weight)

  3.        Function:             getTensorType<T>        Judge template T belongs to which type in DataType

  4.        Abstract Base Class:  Tensor                  Describe the current tensor runtime information and shape of tensor

  5.        subclass:             TensorWrapper<T>        Contain the specific data of type T

  6.        class:                TensorMap               Used to store the key-value pair of <string, Tensor*>,

*/

#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
#include <cuda_fp16.h>
#include "src/utils/string_utils.h"
#include "src/utils/macro.h"

enum Device
{
  CPU_PINNED,
  CPU,
  GPU
};

enum DataType
{
  FP32,
  FP16,
  INT8,
  INT32,
  BOOL,
  BYTES,
  UNSUPPORTED
};

// 用于获取当前数据类型T的枚举
template <typename T>
DataType getTensorType()
{
  if (std::is_same<T, float>::value || std::is_same<T, const float>::value)
  {
    return DataType::FP32;
  }
  else if (std::is_same<T, half>::value || std::is_same<T, const half>::value)
  {
    return DataType::FP16;
  }
  else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value)
  {
    return DataType::INT8;
  }
  else if (std::is_same<T, int>::value || std::is_same<T, const int>::value)
  {
    return DataType::INT32;
  }
  else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value)
  {
    return DataType::BOOL;
  }
  else if (std::is_same<T, char>::value || std::is_same<T, const char>::value)
  {
    return BYTES;
  }
  else
  {
    return UNSUPPORTED;
  }
}

// 虚基类Tensor，主要存储了该Tensor的运行时信息
template <typename T>
class TensorWrapper;
class Tensor
{
  // 使用一个虚类作为父类，能够让不同类型(typename T)的tensor同时加入同一个字典
  // 这在强类型的c++中是必须的
  // 也因此引出了一个问题,data的类型是被包裹的对象，因此是TensorWrapper中的成员，
  // 也就是说，我们在将Tensor加入字典时，并没有方法访问*data，除非下行转换
public:
  Device location;
  DataType dtype;
  std::vector<int> shape;

  // Constructor
  Tensor() = default;

  Tensor(const Device location_, const DataType dtype_, const std::vector<int> shape_)
      : location(location_), dtype(dtype_), shape(shape_)
  {
  }

  // 使用虚函数获取TensorWrapper元素数量
  virtual int size() const
  {
    if (shape.size() == 0)
    {
      // TODO: add an reminder info
      std::cout << "The Tensor is empty/shape is 0\n"
                << std::endl;
      return 0;
    }
  }

  // 下行转换，用以获得这个Tensor*所指向的具体的TensorWrapper的指针
  // dynamic_cast更安全，此处能够保证Tensor就是Tensorwrapper<T>的类型，
  // 因此可以使用static_cast
  template <typename T>
  TensorWrapper<T> *as()
  {
    // return dynamic_cast<TensorWrapper<T>*>(*this);
    return static_cast<TensorWrapper<T> *>(*this);
  }

  // Debug函数，用以返回包含相关信息的字符串
  std::string DeviceString() const
  {
    constexpr static const std::unordered_map<Device, std::string> devicestring{
        {Device::CPU, "CPU"},
        {Device::CPU_PINNED, "CPU_PINNED"},
        {Device::GPU, "GPU"}};

    return devicestring.at(location);
  }

  // Debug函数，返回所有相关信息
  virtual std::string toString() const
  {
    std::string device_info = DeviceString();

    constexpr static const std::unordered_map<DataType, std::string> type_to_string{
        {DataType::INT8, "INT8"},
        {DataType::INT32, "INT32"},
        {DataType::FP16, "FP16"},
        {DataType::FP32, "FP32"}};

    return fmtstr("Tensor[where = %s, type = %s, shape = %s]",
                  device_str.c_str(),
                  type_to_string.at(dtype).c_str(),
                  vec2str(shape), c_str());
  }

  // 检查data是否为空
  virtual bool data_is_null() const = 0;
};

// 子类TensorWrapper，用以保存数据。
template <typename T>
class TensorWrapper : public Tensor
{
public:
  T *data;

  // Constructor
  TensorWrapper(Device location, DataType dtype, std::vector<int> shape)
      : Tensor(location, dtype, shape)
  {
  }

  TensorWrapper(Device location, DataType dtype, std::vector<int> shape, T *data_)
      : Tensor(location, dtype, shape), data(data_)
  {
    DataType in_dtype = getTensorType<T>();

    // 检查传入参数类型应该等于参数数据类型
    LLM_CHECK_WITH_INFO(in_dtype == dtype, "the passed in data type <T> should be as same as dtype in params");
  }

  // 获取数据形状
  virtual int size() override const
  {
    if (data == nullptr || shape.size() == 0)
    {
      // TODO: 提示这个数据有问题
      return 0;
    }
    // 返回shape数据相乘(iterator begin,iterator end, initial value, operator)
    return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
  }

  // 获取CPU索引对应数据（注意[]操作符只能访问CPU数据）
  inline T getVal(int id) const
  {
    // TODO: need boundry check and device check
    LLM_CHECK(location == CPU);
    return data[id];
  }

  // 获取CPU中data第一个数据
  inline T getVal() const
  {
    // TODO: need type check
    LLM_CHECK(location == CPU);
    return getVal(0);
  }

  // 获取这个数据的指针
  inline T *getPtr() const
  {
    return (T *)data;
  }

  // Debug选项
  virtual std::string toString() override const
  {
    std::string device_str = DeviceString();

    static const std::unordered_map<DataType, std::string> type_to_string{
        {DataType::INT8, "INT8"},
        {DataType::FP16, "FP16"},
        {DataType::FP32, "FP32"}};

    return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
                  device_str.c_str(),
                  type_to_string.at(dtype).c_str(),
                  vec2str(shape).c_str(),
                  data);
  }

  // 检查data是否为空
  virtual bool data_is_null() override const
  {
    return data == nullptr;
  }
};




#include<initializer_list>
//
class TensorMap
{
public:
    std::unordered_map<std::string, Tensor*> tensor_map_;

    TensorMap() = default;

    //std::initializer_list<>支持类似TensorMap({{},{}})这样传参
    TensorMap(std::initializer_list<std::pair<std::string, Tensor*>> tensor_map){
        for(auto& pair:tensor_map){
            if(pair.second->data_is_null() == false){
                insert(pair.first, pair.second);
            }
            else{
                std::cout << "this is not a valid tensor, skip to insert into tensormap" << std::endl;
            }
        }
    }

    ~TensorMap(){
        tensor_map_.clear();
    }

    inline size_t size() const{
        return tensor_map_.size();
    }

    inline bool isExist(const std::string& key) const{
        return tensor_map_.find(key) != tensor_map_.end();
    }

    inline bool isValid(const Tensor* tensor){
        return tensor->size() > 0;
    }

    // add
    inline void insert(const std::string& key, Tensor* value){
        tensor_map_[key] = value;
    }

    inline void insert(std::pair<std::string,Tensor*> p){
        tensor_map_.insert(p);
    }

    // find
    inline Tensor* at(const std::string& key){
        if(isExist(key)){
            return tensor_map_.at(key);
        }
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor* operator[](const std::string& key)
    {
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map    (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);

    }
    // change

    // delete

    // Debug, 输出tensormap所有的string
    std::vector<std::string> keys() const{
        std::vector<std::string> key_names;
        for(auto& kv : tensor_map_){
            key_names.push_back(kv.first);
        }
        return key_names;
    }

    //打印tensormap所有的key
    std::string toString()
    {
        std::stringstream ss;
        ss << "{";
        std::vector<std::string> key_names = keys();
        for (size_t i = 0; i < tensor_map_.size(); ++i) {
            ss << key_names[i] << ": " << at(key_names[i])->toString();
            if (i < tensor_map_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }
}
