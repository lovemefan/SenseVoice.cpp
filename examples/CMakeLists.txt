add_library(common STATIC
        common-ggml.h
        common-ggml.cc
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common ggml)
add_subdirectory(quantize)