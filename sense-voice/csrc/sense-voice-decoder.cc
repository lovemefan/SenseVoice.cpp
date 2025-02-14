//
// Created by lovemefan on 2024/7/25.
//

#include "sense-voice-decoder.h"

#define SENSEVOICE_DECODER_MAX_NODES 16

// faster matrix multiplications for tensors that do not have dimension 0 divisible by "pad"
// the idea is to represent the original matrix multiplication:
//
//   Z = X @ Y
//
// with the sum of two matrix multiplications:
//
//   Z = (X_0 @ Y_0) + (X_1 @ Y_1)
//
// here X_0 and Y_0 are views of X and Y that have dimension 0 divisible by "pad"
// and X_1 and Y_1 are the remaining views. X_1 and Y_1 end up being small matrices that can be processed with more
// general-purpose kernels
//
static struct ggml_tensor * ggml_mul_mat_pad(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y, int pad = 32) {
    // use padding only if dimension 0 is at least 8 times larger than the padding
    // else we won't get much benefit from the optimization
    const int n_pad_req = 8;

    if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) {
        return ggml_mul_mat(ctx, x, y);
    }

    struct ggml_tensor * x_0 = ggml_view_3d(ctx, x, (x->ne[0]/pad)*pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    struct ggml_tensor * x_1 = ggml_view_3d(ctx, x,  x->ne[0]%pad,      x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_0->ne[0]*x_0->nb[0]);

    struct ggml_tensor * y_0 = ggml_view_3d(ctx, y, (y->ne[0]/pad)*pad, y->ne[1], y->ne[2], y->nb[1], y->nb[2], 0);
    struct ggml_tensor * y_1 = ggml_view_3d(ctx, y,  y->ne[0]%pad,      y->ne[1], y->ne[2], y->nb[1], y->nb[2], y_0->ne[0]*y_0->nb[0]);

    return ggml_add(ctx,
                    ggml_mul_mat(ctx, x_0, y_0),
                    ggml_mul_mat(ctx, x_1, y_1));
}

// copy from whisper.cpp
// TODO: CUDA is currently broken - seems ggml_mul_mat does not handle views correctly
#if defined(GGML_USE_METAL)
#define ggml_mul_mat ggml_mul_mat_pad
#endif

struct ggml_cgraph *sense_voice_build_graph_ctc_decoder(sense_voice_context &ctx,
                                                    sense_voice_state &state){
    const auto &model = ctx.model.model;

    struct ggml_init_params params = {
            /*.mem_size   =*/state.sched_decode.meta.size(),
            /*.mem_buffer =*/state.sched_decode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSEVOICE_DECODER_MAX_NODES, false);

    ggml_tensor *encoder_out = ggml_new_tensor_3d(ctx0, state.encoder_out->type,
                                                  state.encoder_out->ne[0], state.encoder_out->ne[1], 
                                                  state.encoder_out->ne[2]);
    ggml_set_name(encoder_out, "encoder_out");
    ggml_set_input(encoder_out);

    ggml_tensor *cur;
    {
        // Reshape encoder_out to merge batch and time dimensions
        cur = ggml_reshape_2d(ctx0, encoder_out, encoder_out->ne[0], encoder_out->ne[1] * encoder_out->ne[2]);
        cur = ggml_mul_mat(ctx0, model->ctc_out_linear_weight, cur);
        cur = ggml_add(ctx0, cur, model->ctc_out_linear_bias);
        // Reshape back to 3D
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], encoder_out->ne[1], encoder_out->ne[2]);
    }
    ggml_tensor * probs = ggml_soft_max(ctx0, cur);
    probs = ggml_reshape_2d(ctx0, probs, probs->ne[0], probs->ne[1] * probs->ne[2] * probs->ne[3]);
    ggml_tensor * argmax_logit = ggml_argmax(ctx0, probs);
    argmax_logit = ggml_reshape_3d(ctx0, argmax_logit, cur->ne[1], cur->ne[2], cur->ne[3]);
    ggml_set_output(probs);
    ggml_set_output(argmax_logit);
    ggml_build_forward_expand(gf, argmax_logit);
    ggml_free(ctx0);
    return gf;
}


bool sense_voice_decode_internal(sense_voice_context &ctx,
                                 sense_voice_state &state,
                                 const int n_threads) {
    const int64_t t_start_us = ggml_time_us();

    // decoder
    {
        auto & sched = state.sched_decode.sched;


        ggml_cgraph *gf = sense_voice_build_graph_ctc_decoder(ctx, state);

//        sched->callback_eval = ctx.params.cb_eval;
//        sched->callback_eval_data = ctx.params.cb_eval_user_data;

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }


        // set the input
        {
            struct ggml_tensor *encoder_out = ggml_graph_get_tensor(gf, "encoder_out");
            ggml_backend_tensor_copy(state.encoder_out, encoder_out);
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
        {
            ggml_tensor *argmax_logit = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
            // TODO 临时处理，建议讨论后取其一
            if(state.result_all.empty()) {
                state.ids.resize(argmax_logit->ne[0]);
                ggml_backend_tensor_get(argmax_logit, state.ids.data(), 0, sizeof(int) * argmax_logit->ne[0]);
            }
            else {
                const int32_t n_logits = argmax_logit->ne[0] * argmax_logit->ne[1];
                // Get the tensor data into a temporary buffer
                std::vector<int> temp_buffer(n_logits);
                ggml_backend_tensor_get(argmax_logit, temp_buffer.data(), 0, sizeof(int) * n_logits);
                for(int32_t i = 0; i < argmax_logit->ne[1]; i++)
                {
                    int posL = i * argmax_logit->ne[0];
                    state.result_all[state.segmentIDs[i]].tokens = std::vector<int>(temp_buffer.begin() + posL, temp_buffer.begin() + posL + argmax_logit->ne[0]);
                }
            }
        }

    }
//    ggml_tensor *logit = ggml_get_tensor(ctx)
    state.t_decode_us += ggml_time_us() - t_start_us;

    return true;
}