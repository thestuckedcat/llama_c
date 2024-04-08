add_library(weights INTERFACE)

target_include_directories(weights INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/llama_weight)

