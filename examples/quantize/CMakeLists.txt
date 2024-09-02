include_directories(${PROJECT_SOURCE_DIR}/sense-voice/csrc)
add_executable(sense-voice-quantize quantize.cc)
target_link_libraries(sense-voice-quantize PRIVATE  sense-voice-core ggml common)