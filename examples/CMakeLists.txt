add_library(common STATIC
        common-ggml.h
        common-ggml.cc
)

find_package(SDL2 QUIET)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(SDL2_FOUND)
    include_directories(${SDL2_INCLUDE_DIRS}) 
    add_library(streamSDL STATIC common-sdl.h common-sdl.cpp)
    target_link_libraries(common ggml ${SDL2_LIBRARIES} stdc++fs)
    add_subdirectory(stream)
else()
    target_link_libraries(common ggml stdc++fs)
endif()
add_subdirectory(quantize)