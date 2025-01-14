add_library(common STATIC
        common-ggml.h
        common-ggml.cc
)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS}) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(streamSDL STATIC common-sdl.h common-sdl.cpp)
target_link_libraries(common ggml ${SDL2_LIBRARIES} stdc++fs)
add_subdirectory(quantize)

add_subdirectory(stream)