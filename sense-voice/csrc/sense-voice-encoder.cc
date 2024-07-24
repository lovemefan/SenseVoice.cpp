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
#include <map>
#include <string>
#include <vector>



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


struct ggml_tensor *encoder_layer_sanm_forward(sense_voice_hparams hparams, ggml_context *ctx0, ggml_tensor *cur, sense_voice_layer_encoder &layer){

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

        int n_ctx = cur->ne[1];
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

        struct ggml_tensor *fsmn_memory = ggml_new_tensor_2d(ctx0, V->type, V->ne[0], V->ne[1]);

        // conv depth wise 1d with groups=input_channel implement
        {
            fsmn_memory = ggml_conv_depthwise_2d(ctx0,
                                     layer.e_attn_fsmn_w,
                                     ggml_reshape_4d(ctx0, V, 1, V->ne[0], V->ne[1], V->ne[2]),
                                     1,1,0,0,0,0);
            fsmn_memory = ggml_reshape_3d(ctx0, fsmn_memory, V->ne[0], V->ne[1], V->ne[2]);
            fsmn_memory = ggml_add(ctx0, fsmn_memory, V);
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
    return cur;

}

struct ggml_cgraph *sense_voice_build_graph_encoder(sense_voice_context &pctx,
                                                   sense_voice_state &pstate) {
    const auto &model = pctx.model.model;
    const auto &hparams = pctx.model.hparams;
    const int n_ctx =
            pstate.exp_n_audio_ctx > 0 ? pstate.exp_n_audio_ctx : hparams.n_audio_ctx;

    struct ggml_init_params params = {
            /*.mem_size   =*/pstate.alloc_encode.meta.size(),
            /*.mem_buffer =*/pstate.alloc_encode.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSEVOICE_MAX_NODES, false);

    struct ggml_tensor *fbank = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_F32, hparams.n_mels * hparams.lfr_m, n_ctx);

    struct ggml_tensor *embedding_position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4);

    ggml_set_name(fbank, "fbank");
    ggml_set_input(fbank);


    ggml_tensor *embedding = ggml_get_rows(ctx0, model->embedding, embedding_position);

    struct ggml_tensor *cur = ggml_concat(ctx0, embedding, fbank, 1);

    cur = ggml_scale(ctx0, cur, sqrtf(hparams.n_encoder_hidden_state));

    // todo : implement encoder small forward graph
    // [x] 1. sinusoidal position
    // [x] 2. encoders0
    // [x] 3. encoders
    // [x] 4. tp_encoders
    // [ ] 5. tp_norm
    // [ ] 6. linear
    ggml_tensor *position = ggml_new_tensor_2d(ctx0, cur->type, cur->ne[0], cur->ne[1]);

    // construct position embedding
    {
        auto n_len = cur->ne[1];
        auto dim = fbank->ne[0];
        std::vector<float> _position;
        _position.resize(n_len * dim);

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

        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_reg_get_default_buffer_type(cur->type), _position.size() * sizeof(float));
        struct ggml_tallocr allocr = ggml_tallocr_new(buffer);
        ggml_tallocr_alloc(&allocr, position);

        ggml_backend_tensor_set(
                position, _position.data(), 0,
                ggml_nelements(position) * sizeof(float));
    }

    cur = ggml_add(ctx0, position, cur);

    // encoders0 forward
    cur = encoder_layer_sanm_forward(hparams, ctx0, cur, model->encoder->encoder0);

    // encoders forward
    for (int i=0; i < hparams.n_encoder_layers - 1; i++){
        cur = encoder_layer_sanm_forward(hparams, ctx0, cur, model->encoder->encoders_layer[i]);
    }
    // tp encoders forward
    for (int i=0; i < hparams.n_tp_encoder_layers; i++){
        cur = encoder_layer_sanm_forward(hparams, ctx0, cur, model->encoder->tp_encoders_layer[i]);
    }

    cur = ggml_transpose(ctx0, cur);



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

bool set_sense_voice_encoder_layer_sanm(std::vector<sense_voice_layer_encoder> &encoder,
                                        std::map<std::string,
                                        struct ggml_tensor *> &tensors,
                                        int n_encoder_layers,
                                        std::string prefix){
    for (int i = 0; i < n_encoder_layers; ++i) {
        auto layer = &encoder[i];
        // map by name
        layer->e_attn_ln_out_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_out.weight"];
        layer->e_attn_ln_out_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_out.bias"];

        layer->e_attn_ln_qkv_w =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_q_k_v.weight"];
        layer->e_attn_ln_qkv_b =
                tensors["encoder." + prefix + "." + std::to_string(i) +
                        ".self_attn.linear_q_k_v.bias"];

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
