add_library(kernels STATIC 
            build_causal_mask.cu
            cublas_utils.cu 
            input_embedding.cu 
            linear.cu 
            paddingoffset.cu 
            rmsnorm.cu)
target_link_libraries(kernels PUBLIC utils weights)
target_include_directories(kernels PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR})        

