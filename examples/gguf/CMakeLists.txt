add_executable(gguf gguf.cc)
install(gguf RUNTIME)
target_link_libraries(gguf PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(gguf PRIVATE cxx_std_11)