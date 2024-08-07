cmake_minimum_required(VERSION 3.12)
project(SenseVoice.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -pthread")

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

option(SENSEVOICE_BUILD_TESTS "sense-voice: build tests" OFF)

option(GGML_USE_BLAS                     "usr blas"  OFF)
option(SENSE_VOICE_COREML                "sense voice: enable Core ML framework"  OFF)
option(SENSE_VOICE_COREML_ALLOW_FALLBACK "sense voice: allow non-CoreML fallback" OFF)
option(SENSE_VOICE_OPENVINO              "sense voice: support for OpenVINO"      OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()


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

