add_library(common STATIC
        common-ggml.h
        common-ggml.cc
)

find_package(SDL2 QUIET)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(ExampleLibs ggml)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3)
    list(APPEND ExampleLibs stdc++fs)
endif()
if(SDL2_FOUND)
    include_directories(${SDL2_INCLUDE_DIRS}) 
    add_library(streamSDL STATIC common-sdl.h common-sdl.cpp)
    list(APPEND ExampleLibs ${SDL2_LIBRARIES})
endif()
target_link_libraries(common ${ExampleLibs})
add_subdirectory(quantize)
if(SDL2_FOUND)
    add_subdirectory(stream)
endif()
