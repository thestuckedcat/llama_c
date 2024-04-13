add_executable(linearTest linear_test.cu)
target_link_libraries(linearTest kernels utils weights)

# 添加Openblas库
# set(OpenBLAS_DIR "D:/Openblas")
# 添加头文件目录
# target_include_directories(linearTest PRIVATE ${OpenBLAS_DIR}/include)

#寻找并添加库
# find_library(OpenBLAS_LIBRARIES 
#                 NAMES libopenblas 
#                 PATHS ${OpenBLAS_DIR}/lib)
# if(NOT OpenBLAS_LIBRARIES){
#     message(FATAL_ERROR "OpenBLAS library not found")
# }

# target_link_libraries(linearTest ${OpenBLAS_LIBRARIES})
