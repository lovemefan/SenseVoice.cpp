//
// Created by lovemefan on 2024/7/19.
//

#ifndef SENSEVOICE_CPP_SENSE_VOICE_ENCODER_H
#define SENSEVOICE_CPP_SENSE_VOICE_ENCODER_H


#include <ggml.h>
#include "common.h"


// Progress callback
typedef void (*sense_voice_progress_callback)(struct sense_voice_context *ctx,
                                             struct sense_voice_state *state,
                                             int progress, void *user_data);



// Various functions for loading a ggml sense_voice model.
// Allocate (almost) all memory needed for the model.
// Return NULL on failure

SENSEVOICE_API struct sense_voice_context_params;


SENSEVOICE_API struct ggml_cgraph *sense_voice_build_graph_encoder(
        sense_voice_context &wctx, sense_voice_state &wstate);

// Frees all allocated memory
SENSEVOICE_API void sense_voice_free(struct sense_voice_context *ctx);
SENSEVOICE_API void sense_voice_free_params(
        struct sense_voice_full_params *params);


#endif//SENSEVOICE_CPP_SENSE_VOICE_ENCODER_H
