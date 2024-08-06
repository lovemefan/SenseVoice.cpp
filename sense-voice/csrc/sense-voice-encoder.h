//
// Created by lovemefan on 2024/7/19.
//

#ifndef SENSEVOICE_CPP_SENSE_VOICE_ENCODER_H
#define SENSEVOICE_CPP_SENSE_VOICE_ENCODER_H


#include <ggml.h>
#include "common.h"


// ############ model structure #############

struct sense_voice_layer_encoder {
    // encoder_attn.linear_out.weight
    struct ggml_tensor *e_attn_ln_out_w;
    struct ggml_tensor *e_attn_ln_out_b;

    // encoder.self_attn.linear_q_k_v.weight
    struct ggml_tensor *e_attn_ln_q_w;
    struct ggml_tensor *e_attn_ln_q_b;

    struct ggml_tensor *e_attn_ln_k_w;
    struct ggml_tensor *e_attn_ln_k_b;

    struct ggml_tensor *e_attn_ln_v_w;
    struct ggml_tensor *e_attn_ln_v_b;

    // encoder.self_attn.fsmn_block.weight
    struct ggml_tensor *e_attn_fsmn_w;

    // encoder.feed_forward.w_1.weight
    struct ggml_tensor *e_mlp_w1;
    struct ggml_tensor *e_mlp_b1;

    // encoder.feed_forward.w_2.weight
    struct ggml_tensor *e_mlp_w2;
    struct ggml_tensor *e_mlp_b2;

    // encoder.norm1.weight
    struct ggml_tensor *e_norm_w1;
    struct ggml_tensor *e_norm_b1;

    // encoder.norm2.weight
    struct ggml_tensor *e_norm_w2;
    struct ggml_tensor *e_norm_b2;
};

struct sense_voice_encoder {
    ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    ggml_type itype =
            ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)
    sense_voice_layer_encoder encoder0;
    std::vector<sense_voice_layer_encoder> encoders_layer;
    std::vector<sense_voice_layer_encoder> tp_encoders_layer;

    // encoder.tp_norm.weight
    struct ggml_tensor *e_tp_norm_w;
    struct ggml_tensor *e_tp_norm_b;

    // encoder.after_norm.weight
    struct ggml_tensor *e_after_norm_w;
    struct ggml_tensor *e_after_norm_b;
};


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

bool set_sense_voice_encoder_layer_sanm(
        std::vector<sense_voice_layer_encoder> &layer, std::map<std::string,
        struct ggml_tensor *> &tensors, int n_encoder_layers, const std::string &prefix);

bool sense_voice_encode_internal(sense_voice_context &ctx,
                            sense_voice_state &state,
                            const int n_threads);

#endif//SENSEVOICE_CPP_SENSE_VOICE_ENCODER_H
