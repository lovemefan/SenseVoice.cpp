//
// Created by lovemefan on 2024/7/25.
//

#include "sense-voice-decoder.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#define SENSEVOICE_DECODER_MAX_NODES 8
struct ggml_cgraph *sense_voice_build_graph_ctc_decoder(sense_voice_context &ctx,
                                                    sense_voice_state &state){
    const auto &model = ctx.model.model;
    const auto &hparams = ctx.model.hparams;

    struct ggml_init_params params = {
            /*.mem_size   =*/state.sched_decode.meta.size(),
            /*.mem_buffer =*/state.sched_decode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSEVOICE_DECODER_MAX_NODES, false);

    ggml_tensor *encoder_out = ggml_new_tensor_2d(ctx0, state.encoder_out->type,
                                                  state.encoder_out->ne[0], state.encoder_out->ne[1]);
    ggml_set_name(encoder_out, "encoder_out");
    ggml_set_input(encoder_out);

    ggml_tensor *cur;
    {
        cur = ggml_mul_mat(ctx0, model->ctc_out_linear_weight, encoder_out);
        cur = ggml_add(ctx0, cur, model->ctc_out_linear_bias);
    }
    ggml_tensor * probs = ggml_soft_max(ctx0, cur);
    ggml_tensor * argmax_logit = ggml_argmax(ctx0, probs);
    ggml_set_output(probs);
    ggml_set_output(argmax_logit);
    ggml_build_forward_expand(gf, argmax_logit);
    return gf;
}

static bool ggml_graph_compute_helper(
        ggml_backend_sched_t   sched,
        struct ggml_cgraph * graph,
        int   n_threads) {

    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
        //#ifdef GGML_USE_BLAS
        //        if (ggml_backend_is_blas(backend)) {
        //            ggml_backend_blas_set_n_threads(backend, n_threads);
        //        }
        //#endif

#ifdef GGML_USE_METAL
        if (ggml_backend_is_metal(backend)) {
            ggml_backend_metal_set_n_cb(backend, n_threads);
        }
#endif
    }

    bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_reset(sched);
    return t;
}

bool sense_voice_decode_internal(sense_voice_context &ctx,
                                 sense_voice_state &state,
                                 const int n_threads) {
    const int64_t t_start_us = ggml_time_us();

    const auto &model = ctx.model;
    const auto &hparams = model.hparams;


    // decoder
    {
        auto & sched = state.sched_decode.sched;

        ggml_cgraph *gf = sense_voice_build_graph_ctc_decoder(ctx, state);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }


        // set the input
        {
            struct ggml_tensor *encoder_out = ggml_graph_get_tensor(gf, "encoder_out");
            ggml_backend_tensor_set(
                    encoder_out, state.encoder_out->data, 0,
                    ggml_nelements(encoder_out) * sizeof(float));
        }


        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }

        {
            ggml_tensor *argmax_logit = gf->nodes[gf->n_nodes - 1];
            state.ids = std::vector<int>((int *)argmax_logit->data, (int *)argmax_logit->data + argmax_logit->ne[0]);

            for(int id: state.ids){
                if (id != 0) {
                    printf("%s", ctx.vocab.id_to_token[id].c_str());
                }
            }
        }

    }
//    ggml_tensor *logit = ggml_get_tensor(ctx)
    state.t_decode_us += ggml_time_us() - t_start_us;

    return true;
}