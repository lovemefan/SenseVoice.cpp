//
// Created by lovemefan on 2024/11/24.
//

#ifndef SENSEVOICE_CPP_SILERO_VAD_H
#define SENSEVOICE_CPP_SILERO_VAD_H


#include <ggml.h>
#include "common.h"


struct silero_vad_stft {
    struct ggml_tensor *forward_basis_buffer;
};

struct silero_vad_encoder_layer {
    // conv1d
    struct ggml_tensor *reparam_conv_w;
    struct ggml_tensor *reparam_conv_b;
};


struct silero_vad_decoder {
    
    // lstm cell
    struct ggml_tensor *lstm_weight_ih;
    struct ggml_tensor *lstm_bias_ih;
    struct ggml_tensor *lstm_weight_hh;
    struct ggml_tensor *lstm_bias_hh;

    // conv1d
    struct ggml_tensor * decoder_conv_w;
    struct ggml_tensor * decoder_conv_b;

};

struct silero_vad {
    silero_vad_stft stft;
    std::vector<silero_vad_encoder_layer> encoders_layer;
    silero_vad_decoder decoder;
};

// Progress callback
typedef void (*silero_vad_progress_callback)(struct silero_vad_context *ctx,
                                              struct silero_vad_state *state,
                                              int progress, void *user_data);



// Various functions for loading a ggml silero_vad model.
// Allocate (almost) all memory needed for the model.
// Return NULL on failure

SENSEVOICE_API struct silero_vad_context_params;


SENSEVOICE_API struct ggml_cgraph *silero_vad_build_graph(
        sense_voice_context &ctx, sense_voice_state &state);


SENSEVOICE_API bool silero_vad_encode_internal(sense_voice_context &ctx,
                                               sense_voice_state &state,
                                               std::vector<float> chunk,
                                               const int n_threads,
                                               float &speech_prob);

SENSEVOICE_API double silero_vad_with_state(sense_voice_context &ctx,
                           sense_voice_state &state,
                           std::vector<float> &pcmf32,
                           int n_processors);

#endif//SENSEVOICE_CPP_SILERO_VAD_H
