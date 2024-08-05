//
// Created by lovemefan on 2024/7/25.
//

#ifndef SENSEVOICE_CPP_SENSE_VOICE_DECODER_H
#define SENSEVOICE_CPP_SENSE_VOICE_DECODER_H

#include <ggml.h>
#include "common.h"
struct ggml_cgraph *sense_voice_build_graph_ctc_decoder(sense_voice_context &ctx,
                                                        sense_voice_state &state);
bool sense_voice_decode_internal(sense_voice_context &ctx,
                                 sense_voice_state &state,
                                 const int n_threads);
#endif//SENSEVOICE_CPP_SENSE_VOICE_DECODER_H
