# 版本控制

cmake_minimum_required(VERSION 3.8 FATAL_ERROR) 
    # FATAL_ERROR:版本过低时直接停止运行。
project(llama LANGUAGES CXX CUDA)
    # LANGUAGES:指定语言为C++与CUDA


# 索引CUDA库，并将根目录存为PATH以方便以后调用
find_package(CUDA 10.0 REQUIRED)
    # REQUIRED:必须找到包，否则报错

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
    # CMAKE_MODULE_PATH:用于指定CMake查找模块的额外路径，此处添加CUDA_PATH/lib64到查找路径


#[[
    设置编译器标志
]]


# 固有字段CMAKE_C_FLAGS,CMAKE_CXX_FLAGS,CMAKE_CUDA_FLAGS

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS   "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
# -Xcompilor:允许向底层编译器传递选项。
# -Wall     :启用编译器警告

# 为不同的架构生成二进制代码
#[[set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
]]
# 以上在CMake3.18以后可以被简化为
# 这样设置后，CMake会自动为这些架构添加适当的-gencode选项，而无需手动指定每个架构的-gencode参数。
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# 在CMake配置过程中输出一条消息，通知用户已指定的GPU架构。
message("-- Assign GPU architecture (sm=70 75 80 86)")

#[[ -Wall开启所有警告，
    -O0禁用优化，这有助于调试。
    对于CUDA，
    -G生成调试信息，
    -Xcompiler将-Wall标志传递给底层的C/C++编译器。]]
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")


#[[
    STATUS 关键字意味着这条信息将以一种非侵入式的方式显示给用户，通常前缀为--，用于区分普通消息。

    "CMAKE_CXX_FLAGS" 是文本字符串，直接显示在消息中，表示接下来显示的是关于C++编译器标志的信息。

    ${CMAKE_CXX_FLAGS} 是CMake变量，包含了当前为C++编译器设置的所有标志。

    通常会显示为:
    -- CMAKE_CXX_FLAGS: -O2 -g -Wall

]]
message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})






#[[
    设置C++标准和要求:
    设置C++11为标准
    设置如果C++11不可用，配置失败
]]

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#[[
    如果项目是C++11作为标准，为CUDA编译器设定执行编译标志:

    STREQUAL: 字符串比较，判断两个字符串是否相同


    --expt-extended-lambda：

    这个标志启用CUDA中对扩展的lambda表达式的支持。C++11引入了lambda表达式，这是一个强大的特性，允许在代码中使用匿名函数。在CUDA程序中，使用扩展的lambda可以更方便地编写GPU上的并行算法。这个标志特别有用，因为它允许在CUDA核函数（kernel functions）中直接使用lambda表达式。


    --expt-relaxed-constexpr：

    这个标志放宽了对constexpr表达式的限制。constexpr是C++11中引入的一个关键字，用于在编译时计算表达式的值。在CUDA中启用这个标志后，可以在更多的上下文中使用constexpr，提高代码的灵活性和表达能力。

    --std=c++11：

    通过这个标志，明确指定CUDA代码应该按照C++11标准进行编译。这确保CUDA代码可以使用C++11标准中引入的语言特性和库功能。
]]

if(CMAKE_CXX_STANDARD STREQUAL "11")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()



#[[
    设置C++ RELEASE模式下启用高级优化03

    设置CUDA RELEASE模式下启用03优化

    -Xcompiler 是NVCC特有的一个选项，用于将其后的标志传递给底层的C/C++编译器。这是因为NVCC本身只是一个包装器（wrapper），它会根据需要调用GCC或Clang等C/C++编译器来处理CUDA代码中的C/C++部分。

    通过-Xcompiler -O3，NVCC会将-O3优化标志传递给底层的C/C++编译器，从而确保CUDA代码中的C/C++部分也能获得相应的优化。
]]

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")




#[[
    设置静态库，动态库，可执行文件输出目录:根目录/lib与/bin
]]
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)





#[[
    定义用户可配置选项(PERF,PRINT_DATA,SAVE_DATA)
    然后根据这些选项的状态决定是否向项目的编译命令中添加特定的预处理器定义。

    > PERF：此选项用于控制是否测量模型推理性能。默认情况下，这个功能是关闭的（OFF）。

    > PRINT_DATA：此选项控制是否在调试时打印内核输出。默认情况下，此功能也是关闭的。

    > SAVE_DATA：此选项控制是否保存内核输出以便调试。同样，默认关闭。



    用户可以在运行CMake配置命令时通过添加 -D选项名=ON（如-DPERF=ON）来开启这些选项。

    接下来的if语句检查上述定义的选项是否被设置为ON：

    如果PERF选项开启（即用户设置-DPERF=ON），则通过add_compile_options(-DPERF)添加一个预处理器定义PERF到编译命令中。这样，在C或C++源代码中可以通过#ifdef PERF ... #endif来条件性地编译与性能测量相关的代码块。

    类似地，如果PRINT_DATA选项开启，则添加一个PRINT_DATA定义；如果SAVE_DATA选项开启，则添加一个SAVE_DATA定义。这允许源代码中通过预处理器指令来条件性地编译与打印或保存内核输出相关的代码。


    add_compile_options 命令将提供的编译器标志添加到所有目标（可执行文件和库）的编译命令中，这些目标是在该命令之后定义的。
    也就是说，如果添加-DPERF=ON，所有的都会自动加上-DPERF,对应的代码内的ifdef内容就可以触发

    与target_compile_options不同，target是指定一些文件添加flag

    简单来说，
    * options创建了一个可以通过后缀控制的布尔值
    * 添加-DPERF=ON可以使得默认为OFF的PERF为ON
    * 此时，if判断触发，为所有文件显式定义了一个宏定义(PERF)，可以在文件内用ifdef处理
]]
option (PERF
  "measure model inference performance"
  OFF
)
option (PRINT_DATA
  "print kernel output to debug"
  OFF
)
option (SAVE_DATA
  "save kernel output to debug"
  OFF
)
if (PERF)
    add_compile_options(-DPERF)
endif()
if (PRINT_DATA)
    add_compile_options(-DPRINT_DATA)
endif()
if (SAVE_DATA)
    add_compile_options(-DSAVE_DATA)
endif()

#cmake .. -DPRINT_DATA=ON && make
#cmake .. -DPRINT_DATA=ON -DSAVE_DATA=ON && make
#cmake .. -DPERF=ON && make
#cmake .. && make






#[[
    主体部分
]]

#[[
    处理包含目录: 确保编译阶段，源代码引用的所有<<头文件>>都能被编译器找到(将头文件插入目标文件并生成对象文件)
    
    指定编译器搜索头文件（.h或.hpp）的目录。当源文件中使用#include预处理指令引用头文件时，编译器会在这些包含目录中查找指定的头文件。

    如果你的项目使用了第三方库，你需要将该库的头文件目录添加到包含目录中，以便编译器能够找到并正确处理源代码中的#include指令。


    处理链接目录: 将<<编译生成的一个或多个对象文件与必要的库文件链接在一起>>，生成最终的可执行文件或库文件链接阶段，确保链接器能够找到程序依赖的所有外部<<库文件>>。

    指定链接器搜索库文件（如.lib、.a或.so文件）的目录。在生成可执行文件或动态库时，如果程序使用了外部库的函数或对象，链接器需要找到这些库文件以解决外部符号的引用。
]]

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})



#[[
    自动收集所有源文件
]]
file(GLOB_RECURSE LLM_CXX_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE LLM_CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu)

# 为所有源文件添加为llmengine库
add_library(llmengine OBJECT
           ${LLM_CXX_SOURCES}
           ${LLM_CUDA_SOURCES}
           )


# add_subdirectory(src)
# add_subdirectory(tests)
# add_subdirectory(examples)

add_executable(main tests/paddingoffsettest/padding_offset_test.cu)
target_link_libraries(main PUBLIC -lcublas -lcudart -lcudadevrt llmengine)














    

