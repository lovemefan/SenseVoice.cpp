include_directories(${PROJECT_SOURCE_DIR}/sense-voice/csrc)
add_executable(sense-voice-zcr-main main.cc)
target_link_libraries(sense-voice-zcr-main PRIVATE sense-voice-core ggml common)