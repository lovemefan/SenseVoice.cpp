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
target_link_libraries(sense-voice-core PRIVATE ggml stdc++fs)