add_executable(linearTest linear_test.cu)
target_link_libraries(linearTest kernels utils weights)