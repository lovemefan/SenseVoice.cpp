set(SOURCE_FILES
        common.h
        common.cc
        sense-voice-frontend.h
        sense-voice-frontend.cc
        fftsg.cc
        log-mel-filter-bank.h
        sense-voice-encoder.h
        sense-voice-encoder.cc
        sense-voice-decoder.h
        sense-voice-decoder.cc
        sense-voice.h
        sense-voice.cc
        silero-vad.h
        silero-vad.cc
)

add_library(sense-voice-core STATIC ${SOURCE_FILES})

set(ExampleLibs ggml)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    list(APPEND ExampleLibs stdc++fs)
endif()

target_link_libraries(sense-voice-core PRIVATE ${ExampleLibs})