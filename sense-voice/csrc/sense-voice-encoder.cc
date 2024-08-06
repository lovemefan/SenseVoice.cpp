//
// Created by lovemefan on 2024/7/19.
//

#include "sense-voice-encoder.h"

#include <cmath>
#include <ggml.h>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "whisper-mel-cuda.hpp"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

//#ifdef GGML_USE_BLAS
//#include <ggml-blas.h>
//#endif

#include <cassert>
#include <map>
#include <string>
#include <vector>
#define SENSE_VOICE_ENCODER_MAX_NODES 4096


static const size_t MB = 1ull * 1024 * 1024;


struct sense_voice_context *sense_voice_init(struct gguf_context *g_context) {
    ggml_time_init();

    struct sense_voice_context *ctx = new sense_voice_context;

    return ctx;
}

struct sense_voice_context_params sense_voice_context_default_params() {
    struct sense_voice_context_params result = {
            /*.use_gpu              =*/ true,
            /*.flash_attn           =*/ false,
            /*.gpu_device           =*/ 0
    };
    return result;
}

static ggml_backend_t sense_voice_backend_init(
        const sense_voice_context_params &params) {
    ggml_backend_t backend_gpu = nullptr;

    // initialize the backends
#ifdef GGML_USE_CUDA
    if (params.use_gpu) {
        SENSE_VOICE_LOG_INFO("%s: using CUDA backend\n", __func__);
        backend_gpu = ggml_backend_cuda_init(params.gpu_device);
        if (!backend_gpu) {
            SENSE_VOICE_LOG_ERROR("%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (params.use_gpu) {
        SENSEVOICE_LOG_INFO("%s: using Metal backend\n", __func__);
        ggml_backend_metal_log_set_callback(g_state.log_callback,
                                            g_state.log_callback_user_data);
        backend_gpu = ggml_backend_metal_init();
        if (!backend_gpu) {
            SENSEVOICE_LOG_ERROR("%s: ggml_backend_metal_init() failed\n", __func__);
        } else if (!ggml_backend_metal_supports_family(backend_gpu, 7)) {
            SENSEVOICE_LOG_ERROR(
                    "%s: Metal GPU does not support family 7 - falling back to CPU\n",
                    __func__);
            ggml_backend_free(backend_gpu);
            backend_gpu = nullptr;
        }
    }
#endif

#ifdef GGML_USE_SYCL
    if (params.use_gpu) {
        SENSE_VOICE_LOG_INFO("%s: using SYCL backend\n", __func__);
        backend_gpu = ggml_backend_sycl_init(params.gpu_device);
        if (!backend_gpu) {
            SENSE_VOICE_LOG_ERROR("%s: ggml_backend_sycl_init() failed\n", __func__);
        }
    }
#endif

    if (backend_gpu) {
        return backend_gpu;
    }
    return ggml_backend_cpu_init();
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

struct ggml_tensor *encoder_layer_sanm_forward(const sense_voice_hparams &hparams,
                                               ggml_context *ctx0,
                                               ggml_tensor *cur,
                                               sense_voice_layer_encoder &layer,
                                               bool user_flash_attn){

    const int n_state = hparams.n_encoder_hidden_state;
    const int n_head = hparams.n_encoder_attention_heads;

    struct ggml_tensor *residual = nullptr;

    if (layer.e_norm_w1->ne[0] == layer.e_norm_w2->ne[0]) {
        residual = ggml_cpy(
                ctx0, cur,
                ggml_new_tensor_2d(ctx0, cur->type, cur->ne[0], cur->ne[1]));
    }

    {
        // layer norm
        // cur = ln_0_w*cur + ln_0_b
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur =
                ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w1), layer.e_norm_b1);
    }

    // self attention
    {
        // self attention linear qkv
        //      cur = ggml_transpose(ctx0, cur);
        // split qkv into separate tensors
        // q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        //  ref:
        //  https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/attention.py#L391-L396
        struct ggml_tensor *Q;
        struct ggml_tensor *Q_h;
        struct ggml_tensor *K;
        struct ggml_tensor *K_h;
        struct ggml_tensor *V;
        struct ggml_tensor *V_h;

        int n_ctx = cur->ne[1];

        Q = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, layer.e_attn_ln_q_w, cur),
                     layer.e_attn_ln_q_b);

        Q_h = ggml_reshape_3d(ctx0, Q, n_state / n_head, n_head, n_ctx);
        Q_h = ggml_permute(ctx0, Q_h,  0, 2, 1, 3);
        Q_h = ggml_cont(ctx0, Q_h);

        ggml_set_name(Q_h, "attention_Q");

        K = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, layer.e_attn_ln_k_w, cur),
                     layer.e_attn_ln_k_b);

        K_h = ggml_reshape_3d(ctx0, K, n_state / n_head, n_head, n_ctx);
        K_h = ggml_permute(ctx0, K_h, 0, 2, 1, 3);
        K_h = ggml_cont(ctx0, K_h);
        //      K = ggml_reshape_3d(ctx0, K, n_state, n_ctx, n_head);
        ggml_set_name(K_h, "attention_K");

        V = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, layer.e_attn_ln_v_w, cur),
                     layer.e_attn_ln_v_b);
        ggml_set_name(V, "attention_V");

        V_h = ggml_reshape_3d(ctx0, V, n_state / n_head, n_head, n_ctx);
        V_h = ggml_permute(ctx0, V_h, 0, 2, 1, 3);
        V_h = ggml_cont(ctx0, V_h);

        // fsmn forward with V
        int padding = (hparams.fsmn_kernel_size - 1) / 2;


        struct ggml_tensor *fsmn_memory = ggml_new_tensor_2d(ctx0, V->type, V->ne[0], V->ne[1]);

        // conv depth wise
        {
            {
                // implement conv depth wise with groups=input_channel implement
                // same in pytorch : F.conv1d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=n_state)
                struct ggml_tensor * a = layer.e_attn_fsmn_w;
                struct ggml_tensor * b = ggml_cont(ctx0, ggml_transpose(ctx0, V));

                struct ggml_tensor * new_a = ggml_reshape_4d(ctx0,
                                                            a,
                                                            a->ne[0],
                                                            1,
                                                            a->ne[1],
                                                            a->ne[2] * a->ne[3]);
                // im2col [n_state, length, kernel_size ]
                struct ggml_tensor * im2col = ggml_im2col(ctx0, new_a,
                                                         ggml_reshape_4d(ctx0, b, b->ne[0], 1, b->ne[1], b->ne[2] * b->ne[3]),
                                                         1, 1, padding, 0, 1, 0, false, GGML_TYPE_F32);


                // new_a [n_state, 1, kernel_size], im2col  [n_state, length, kernel_size]
                // result ->  [n_state, length, kernel_size] @ [n_state, 1, kernel_size].T = [n_state, length , 1]
                struct ggml_tensor * result = ggml_mul_mat(ctx0, new_a, im2col);
                fsmn_memory = ggml_reshape_4d(ctx0, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]);
            }
            fsmn_memory = ggml_cont(ctx0, ggml_transpose(ctx0, fsmn_memory));
            fsmn_memory = ggml_add(ctx0, fsmn_memory, V);
            ggml_set_name(fsmn_memory, "fsmn_memory");
        }

        struct ggml_tensor *KQV;
        float KQscale = 1.0f / sqrtf(float(n_state) / n_head);

        if(user_flash_attn){
            // todo flash attention is not available now
            KQV = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f);
        } else{
            // K * Q
            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K_h, Q_h);

            struct ggml_tensor *KQ_scaled = ggml_scale(ctx0, KQ, KQscale);
            ggml_set_name(KQ_scaled, "score");

            struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_scaled);
            ggml_set_name(KQ_soft_max, "attention");

            KQV = ggml_mul_mat(
                    ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, V_h)), KQ_soft_max);
        }


        struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cpy(ctx0, KQV_merged,
                       ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx));

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.e_attn_ln_out_w, cur),
                       layer.e_attn_ln_out_b);
        ggml_set_name(cur, "attention_out");

        cur = ggml_add(ctx0, cur, fsmn_memory);

        if (layer.e_norm_w1->ne[0] == layer.e_norm_w2->ne[0]) {
            cur = ggml_add(ctx0, cur, residual);
        }
    }

    residual = ggml_cpy(
            ctx0, cur,
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[0], cur->ne[1]));
    {
        // layer norm after attention
        // cur = ln_0_w*cur + ln_0_b
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w2), layer.e_norm_b2);
    }

    {
        // position-wise feed forward layer
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.e_mlp_w1, cur),
                       layer.e_mlp_b1);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.e_mlp_w2, cur),
                       layer.e_mlp_b2);
    }
    // residual after position wise feed forward
    cur = ggml_add(ctx0, cur, residual);
    return cur;

}

struct ggml_cgraph *sense_voice_build_graph_encoder(sense_voice_context &pctx,
                                                   sense_voice_state &pstate) {

    const auto &model = pctx.model.model;
    const auto &hparams = pctx.model.hparams;
    const int n_ctx =
            pstate.exp_n_audio_ctx > 0 ? pstate.exp_n_audio_ctx : hparams.n_audio_ctx;

    struct ggml_init_params params = {
            /*.mem_size   =*/pstate.sched_encode.meta.size(),
            /*.mem_buffer =*/pstate.sched_encode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSE_VOICE_ENCODER_MAX_NODES, false);

    struct ggml_tensor *feature = pstate.feature.tensor;
    ggml_set_name(feature, "feats");
    ggml_set_input(feature);

    struct ggml_tensor *embedding = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4);
    ggml_set_name(embedding, "embedding");
    ggml_set_input(embedding);

    embedding = ggml_get_rows(ctx0, model->embedding, embedding);

    struct ggml_tensor *cur = ggml_concat(ctx0, embedding, feature, 1);

    cur = ggml_scale(ctx0, cur, sqrtf(hparams.n_encoder_hidden_state));

    // implement encoder small forward graph
    //ref: https://github.com/modelscope/FunASR/blob/b7b4a83c18277a7022124cad790c08ae703b7a2d/funasr/models/sense_voice/model.py#L558-L583
    // [x] 1. sinusoidal position
    // [x] 2. encoders0
    // [x] 3. encoders
    // [x] 4. tp_encoders
    // [x] 5. tp_norm
    ggml_tensor *position = ggml_new_tensor_2d(ctx0, cur->type, cur->ne[0], cur->ne[1]);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    cur = ggml_add(ctx0, position, cur);

    // encoders0 forward
    cur = encoder_layer_sanm_forward(hparams, ctx0, cur, model->encoder->encoder0, pctx.params.flash_attn);

    // encoders forward
    for (int i=0; i < hparams.n_encoder_layers - 1; i++){
        cur = encoder_layer_sanm_forward(hparams, ctx0, cur, model->encoder->encoders_layer[i], pctx.params.flash_attn);
    }

    {
        // after encoder norm
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0,
                                      cur,
                                      model->encoder->e_after_norm_w),
                       model->encoder->e_after_norm_b);
    }
    // tp encoders forward
    for (int i=0; i < hparams.n_tp_encoder_layers; i++){
        cur = encoder_layer_sanm_forward(hparams, ctx0, cur, model->encoder->tp_encoders_layer[i], pctx.params.flash_attn);
    }

    {
        // tp encoder norm
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0,
                                      cur,
                                      model->encoder->e_tp_norm_w),
                       model->encoder->e_tp_norm_b);
    }

    ggml_build_forward_expand(gf, cur);

    ggml_set_name(cur, "encoder_out");
    ggml_set_output(cur);
    pstate.encoder_out = cur;
    ggml_free(ctx0);
    return gf;
}

bool sense_voice_encode_internal(sense_voice_context &ctx,
                                       sense_voice_state &state,
                                       const int n_threads) {
    const int64_t t_start_us = ggml_time_us();

    const auto &model = ctx.model;
    const auto &hparams = model.hparams;


    // encoder
    {
        auto & sched = state.sched_encode.sched;

        ggml_cgraph *gf = sense_voice_build_graph_encoder(ctx, state);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }


        // set the inputs

        {
            struct ggml_tensor *position = ggml_graph_get_tensor(gf, "position");
            struct ggml_tensor *embedding = ggml_graph_get_tensor(gf, "embedding");


            auto n_len = position->ne[1];
            auto dim = position->ne[0];
            std::vector<float> _position;
            _position.resize(n_len * dim);

            // construct position embedding
            // sinusoidal position embedding
            // reference:
            // https://github.com/modelscope/FunASR/blob/45d7aa9004763684fb748ee17942ecba81042201/funasr/models/transformer/embedding.py#L392-L405
            // P_{k,i} = sin(k/10000^(2i/d))  0 < i < d/2
            // p_{k,j} = cos(k/10000^(2j/d))  d/2 < j < d

            for (int k = 1; k <= n_len; k++) {
                for (int i = 0; i < dim / 2; i++) {
                    _position[(k - 1) * dim + i] = sinf(k * pow(10000, -2.0 * i / dim));
                    _position[(k - 1) * dim + i + dim / 2] =
                            cosf(k * pow(10000, -2.0 * i / dim));
                }
            }



            ggml_backend_tensor_set(
                    position, _position.data(), 0,
                    ggml_nelements(position) * sizeof(float));



            int _embedding[4];
            _embedding[0] = 0;
            _embedding[1] = 1;
            _embedding[2] = 2;
            _embedding[3] = 14;

            ggml_backend_tensor_set(embedding, &_embedding, 0, 4*sizeof(int));


        }
//        ggml_graph_dump_dot(gf, NULL, "sense-voice.dot");
//        ggml_backend_sched_set_eval_callback(sched, ctx.params.cb_eval, ctx.params.cb_eval_user_data);

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }

    }
    state.t_encode_us += ggml_time_us() - t_start_us;
    return true;
}

bool set_sense_voice_encoder_layer_sanm(std::vector<sense_voice_layer_encoder> &encoder,
                                        std::map<std::string,
                                        struct ggml_tensor *> &tensors,
                                        int n_encoder_layers,
                                        const std::string &prefix){
    for (int i = 0; i < n_encoder_layers; ++i) {
        auto layer = &encoder[i];
        // map by name
        layer->e_attn_ln_out_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_out.weight"];
        layer->e_attn_ln_out_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_out.bias"];

        layer->e_attn_ln_q_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_q.weight"];
        layer->e_attn_ln_q_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_q.bias"];

        layer->e_attn_ln_k_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_k.weight"];
        layer->e_attn_ln_k_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_k.bias"];

        layer->e_attn_ln_v_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_v.weight"];
        layer->e_attn_ln_v_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_v.bias"];

        layer->e_attn_fsmn_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.fsmn_block.weight"];

        layer->e_mlp_w1 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".feed_forward.w_1.weight"];
        layer->e_mlp_b1 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".feed_forward.w_1.bias"];

        layer->e_mlp_w2 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".feed_forward.w_2.weight"];
        layer->e_mlp_b2 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".feed_forward.w_2.bias"];

        layer->e_norm_w1 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".norm1.weight"];
        layer->e_norm_b1 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".norm1.bias"];

        layer->e_norm_w2 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".norm2.weight"];
        layer->e_norm_b2 =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".norm2.bias"];
    }
    return true;
}
