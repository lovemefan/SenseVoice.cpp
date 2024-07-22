//
// Created by lovemefan on 2024/7/19.
//

#include "sense-voice-encoder.h"

#include <math.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "sense-voice-frontend.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdarg.h>

#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>


#define SENSEVOICE_MAX_NODES 4096




static const size_t MB = 1ull * 1024 * 1024;

// ############ model structure #############
struct sense_voice_bias_encoder {
    // bias encoder is a lstm model

    struct ggml_tensor *bias_embed;

    // bias_encoder.weight_ih_l0
    struct ggml_tensor *be_ih_l_w0;
    struct ggml_tensor *be_ih_l_b0;

    // bias_encoder.weight_hh_l0
    struct ggml_tensor *be_hh_l_w0;
    struct ggml_tensor *be_hh_l_b0;

    // bias_encoder.weight_ih_l1
    struct ggml_tensor *be_ih_l_w1;
    struct ggml_tensor *be_ih_l_b1;

    // bias_encoder.weight_hh_l1
    struct ggml_tensor *be_hh_l_w1;
    struct ggml_tensor *be_hh_l_b1;
};

struct sense_voice_layer_encoder {
    // encoder_attn.linear_out.weight
    struct ggml_tensor *e_attn_ln_out_w;
    struct ggml_tensor *e_attn_ln_out_b;

    // encoder.self_attn.linear_q_k_v.weight
    struct ggml_tensor *e_attn_ln_qkv_w;
    struct ggml_tensor *e_attn_ln_qkv_b;

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
    std::vector<sense_voice_layer_encoder> encoder_layer;
    // encoder.after_norm.weight
    struct ggml_tensor *e_after_norm_w;
    struct ggml_tensor *e_after_norm_b;
};


struct sense_voice_context *sense_voice_init(struct gguf_context *g_context) {
    ggml_time_init();

    struct sense_voice_context *ctx = new sense_voice_context;

    return ctx;
}

struct sense_voice_context_params sense_voice_context_default_params() {
    struct sense_voice_context_params result = {
            /*.use_gpu              =*/true,
            /*.gpu_device           =*/0,
    };
    return result;
}

static ggml_backend_t sense_voice_backend_init(
        const sense_voice_context_params &params) {
    ggml_backend_t backend_gpu = NULL;

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
            backend_gpu = NULL;
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


struct ggml_cgraph *sense_voice_build_graph_encoder(sense_voice_context &pctx,
                                                   sense_voice_state &pstate) {
    const auto &model = pctx.model;
    const auto &hparams = model.hparams;
    const int n_ctx =
            pstate.exp_n_audio_ctx > 0 ? pstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_encoder_hidden_state;
    const int n_head = hparams.n_encoder_attention_heads;
    const int n_layer = hparams.n_encoder_layers;

    struct ggml_init_params params = {
            /*.mem_size   =*/pstate.alloc_encode.meta.size(),
            /*.mem_buffer =*/pstate.alloc_encode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSEVOICE_MAX_NODES, false);

    struct ggml_tensor *fbank = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_F32, n_ctx, hparams.n_mels * hparams.lfr_m);

    struct ggml_tensor *position = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_F32, n_ctx, hparams.n_mels * hparams.lfr_m);

    ggml_set_name(fbank, "fbank");
    ggml_set_input(fbank);

    ggml_set_name(position, "position");
    ggml_set_input(position);

    // token encoding + position encoding
    struct ggml_tensor *cur = ggml_add(ctx0, fbank, position);

    static int iter = 0;

    cur = ggml_transpose(ctx0, cur);
    struct ggml_tensor *residual = nullptr;
    for (auto layer : model.encoder->encoder_layer) {
        if (layer.e_norm_w1->ne[0] == layer.e_norm_w2->ne[0]) {
            residual = ggml_cpy(
                    ctx0, cur,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[0], cur->ne[1]));
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
            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.e_attn_ln_qkv_w, cur),
                           layer.e_attn_ln_qkv_b);
            // cur [1536, 1600]
            //      cur = ggml_transpose(ctx0, cur);
            // split qkv into separate tensors
            // q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
            //  ref:
            //  https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/attention.py#L391-L396
            struct ggml_tensor *Q;
            struct ggml_tensor *K;
            struct ggml_tensor *V;
            struct ggml_tensor *V_h;

            Q = ggml_cpy(ctx0,
                         ggml_view_2d(ctx0, cur, n_state, n_ctx, cur->nb[1],
                                      0 * n_state * cur->nb[0]),
                         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx));
            Q = ggml_reshape_4d(ctx0, Q, n_state / n_head, n_head, n_ctx, 1);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            //      Q = ggml_reshape_3d(ctx0, Q, n_state / n_head, n_ctx, n_head);

            K = ggml_cpy(ctx0,
                         ggml_view_2d(ctx0, cur, n_state, n_ctx, cur->nb[1],
                                      1 * n_state * cur->nb[0]),
                         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx));

            K = ggml_reshape_4d(ctx0, K, n_state / n_head, n_head, n_ctx, 1);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            //      K = ggml_reshape_3d(ctx0, K, n_state, n_ctx, n_head);

            V = ggml_cpy(ctx0,
                         ggml_view_2d(ctx0, cur, n_state, n_ctx, cur->nb[1],
                                      2 * n_state * cur->nb[0]),
                         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx));
            V_h = ggml_reshape_4d(ctx0, V, n_state / n_head, n_head, n_ctx, 1);
            V_h = ggml_cont(ctx0, ggml_permute(ctx0, V_h, 0, 2, 1, 3));  // transposed
            //      V = ggml_reshape_3d(ctx0, V, n_state, n_ctx, n_head);

            // fsmn forward with V
            int left_padding = (hparams.fsmn_kernel_size - 1) / 2;
            int right_padding = hparams.fsmn_kernel_size - 1 - left_padding;

            // todo conv depth wise 1d with group implement
            {
                int s0 = 1, s1 = 1, p0 = 0, p1 = 5, d0 = 1, d1 = 1;
                V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
                ggml_conv_1d(ctx0, layer.e_attn_fsmn_w, V, 1, 5, 1);
                struct ggml_tensor *im2col = ggml_im2col(
                        ctx0, layer.e_attn_fsmn_w, V, s0, s1, p0, p1, d0, d1, false,
                        GGML_TYPE_F16);  // [N * IC, OH, OW, KH * KW]
                struct ggml_tensor *new_b =
                        ggml_reshape_4d(ctx0, im2col, im2col->ne[0],
                                        im2col->ne[2] * im2col->ne[1], V->ne[2], V->ne[3]);

                struct ggml_tensor *new_a = ggml_reshape_4d(
                        ctx0, layer.e_attn_fsmn_w,
                        (layer.e_attn_fsmn_w->ne[0] * layer.e_attn_fsmn_w->ne[1]),
                        layer.e_attn_fsmn_w->ne[2], layer.e_attn_fsmn_w->ne[3],
                        1);  // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
                struct ggml_tensor *result = ggml_mul_mat(ctx0, new_a, new_b);

            }

#ifdef USE_FLASH_ATTN

            struct ggml_tensor *KQV = ggml_flash_attn(ctx0, Q, K, V, false);
#else

            // K * Q
            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

            float KQscale = 1.0f / sqrtf(float(n_state) / n_head);

            struct ggml_tensor *KQ_scaled = ggml_scale(ctx0, KQ, KQscale);

            struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_scaled);

            struct ggml_tensor *KQV = ggml_mul_mat(
                    ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, V_h)), KQ_soft_max);
#endif
            struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            cur = ggml_cpy(ctx0, KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx));

            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.e_attn_ln_out_w, cur),
                           layer.e_attn_ln_out_b);

            // todo open when conv depth wise 1d with group implement finished
            // cur = ggml_add(ctx0, cur, fsmn_memory);

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
            cur =
                    ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w2), layer.e_norm_b2);
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
    }

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

static bool sense_voice_encode_internal(sense_voice_context &ctx,
                                       sense_voice_state &state,
                                       const int n_threads) {
    const int64_t t_start_us = ggml_time_us();

    const auto &model = ctx.model;
    const auto &hparams = model.hparams;
    const int n_vocab = hparams.n_vocab;

    auto &logits_out = state.logits;

    struct ggml_tensor *logits;

    // encoder
    {
        auto &alloc = state.alloc_encode.alloc;

        ggml_cgraph *gf = sense_voice_build_graph_encoder(ctx, state);

        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        struct ggml_tensor *fbank = ggml_graph_get_tensor(gf, "fbank");
        struct ggml_tensor *position_embedding =
                ggml_graph_get_tensor(gf, "position");

        // set input
        {
            auto feature = state.feature;
            const int n_ctx = state.feature.n_len;

            assert(fbank->type == GGML_TYPE_F32);
            assert(feature.n_mel == ctx.model.hparams.n_mels);

            fbank->ne[0] = feature.n_len;
            position_embedding->ne[0] = feature.n_len;

            std::vector<float> input(feature.data.begin(), feature.data.end());
            ggml_backend_tensor_set(fbank, input.data(), 0,
                                    ggml_nelements(fbank) * sizeof(float));
        }
        // construct position embedding
        {
            auto n_len = fbank->ne[0];
            auto dim = fbank->ne[1];
            std::vector<float> position;
            position.resize(n_len * dim);

            // sinusoidal position embedding
            // reference:
            // https://github.com/modelscope/FunASR/blob/45d7aa9004763684fb748ee17942ecba81042201/funasr/models/transformer/embedding.py#L392-L405
            // P_{k,i} = sin(k/10000^(2i/d))  0 < i < d/2
            // p_{k,j} = cos(k/10000^(2i/d))  d/2 < j < d

            for (int k = 1; k <= n_len; k++) {
                for (int i = 0; i < dim / 2; i++) {
                    position[(k - 1) * dim + i] = sinf(k * pow(10000, -2.0 * i / dim));
                    position[(k - 1) * dim + i + dim / 2] =
                            cosf(k * pow(10000, -2.0 * i / dim));
                }
            }
            ggml_backend_tensor_set(
                    position_embedding, position.data(), 0,
                    ggml_nelements(position_embedding) * sizeof(float));
        }
    }
    return true;
}

