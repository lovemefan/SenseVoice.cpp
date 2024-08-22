cmake_minimum_required(VERSION 3.12)
project(SenseVoice.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 11)



if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -O3")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -pthread")
endif ()

set(SOURCE_FILES
        sense-voice/csrc/common.h
        sense-voice/csrc/common.cc
        sense-voice/csrc/sense-voice-frontend.h
        sense-voice/csrc/sense-voice-frontend.cc
        sense-voice/csrc/fftsg.cc
        sense-voice/csrc/log-mel-filter-bank.h
        sense-voice/csrc/sense-voice-encoder.h
        sense-voice/csrc/sense-voice-encoder.cc
        sense-voice/csrc/sense-voice-decoder.h
        sense-voice/csrc/sense-voice-decoder.cc
        sense-voice/csrc/sense-voice.h
        sense-voice/csrc/sense-voice.cc
)

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

# general
option(SENSE_VOICE_CCACHE "sense-voice: use ccache if available" ON)

# debug
option(SENSE_VOICE_ALL_WARNINGS           "sense-voice: enable all compiler warnings"                   ON)
option(SENSE_VOICE_ALL_WARNINGS_3RD_PARTY "sense-voice: enable all compiler warnings in 3rd party libs" OFF)

option(SENSE_VOICE_BUILD_TESTS "sense-voice: build tests" OFF)

# sanitizers
option(SENSE_VOICE_SANITIZE_THREAD    "sense-voice: enable thread sanitizer"    OFF)
option(SENSE_VOICE_SANITIZE_ADDRESS   "sense-voice: enable address sanitizer"   OFF)
option(SENSE_VOICE_SANITIZE_UNDEFINED "sense-voice: enable undefined sanitizer" OFF)

# build
option(SENSE_VOICE_FATAL_WARNINGS "sense-voice: enable -Werror flag" OFF)

option(GGML_USE_BLAS                     "usr blas"  OFF)
option(SENSE_VOICE_COREML                "sense voice: enable Core ML framework"  OFF)
option(SENSE_VOICE_COREML_ALLOW_FALLBACK "sense voice: allow non-CoreML fallback" OFF)
option(SENSE_VOICE_OPENVINO              "sense voice: support for OpenVINO"      OFF)

# override ggml options
set(GGML_CCACHE             ${SENSE_VOICE_CCACHE})
set(GGML_SANITIZE_THREAD    ${SENSE_VOICE_SANITIZE_THREAD})
set(GGML_SANITIZE_ADDRESS   ${SENSE_VOICE_SANITIZE_ADDRESS})
set(GGML_SANITIZE_UNDEFINED ${SENSE_VOICE_SANITIZE_UNDEFINED})
set(GGML_ALL_WARNINGS       ${SENSE_VOICE_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS     ${SENSE_VOICE_FATAL_WARNINGS})

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)


if(GGML_USE_BLAS)
    add_definitions(-DGGML_USE_BLAS)
endif()

include_directories(${PROJECT_SOURCE_DIR}/sense-voice/csrc/third-party/ggml/include/ggml ${PROJECT_SOURCE_DIR}/sense-voice/csrc/third-party/ggml/src)
add_subdirectory(sense-voice/csrc/third-party/ggml)
add_library(sense-voice-core STATIC ${SOURCE_FILES})

if (SENSEVOICE_BUILD_TESTS)
    message(DEBUG "Test scripts")
#    add_subdirectory(test)
endif ()


target_link_libraries(sense-voice-core PRIVATE ggml)
add_executable(sense-voice-main sense-voice/csrc/main.cc)
target_link_libraries(sense-voice-main PRIVATE sense-voice-core ggml)

