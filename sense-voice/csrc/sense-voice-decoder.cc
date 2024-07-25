//
// Created by lovemefan on 2024/7/25.
//

#include "sense-voice-decoder.h"
#define SENSEVOICE_DECODER_MAX_NODES 512
struct ggml_cgraph *sense_voice_build_graph_ctc_decoder(sense_voice_context &ctx,
                                                    sense_voice_state &state){
    const auto &model = ctx.model.model;
    const auto &hparams = ctx.model.hparams;

    struct ggml_init_params params = {
            /*.mem_size   =*/state.alloc_decode.meta.size(),
            /*.mem_buffer =*/state.alloc_decode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSEVOICE_DECODER_MAX_NODES, false);

    ggml_tensor *encoder_out = state.encoder_out;
    ggml_tensor *cur;
    {
        cur = ggml_mul_mat(ctx0, model->ctc_out_linear_weight, ggml_transpose(ctx0, encoder_out));
        cur = ggml_add(ctx0, cur, model->ctc_out_linear_bias);
    }
    ggml_set_name(cur, "logit");
    ggml_set_output(cur);

    return gf;
}