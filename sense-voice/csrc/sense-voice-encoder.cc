//
// Created by lovemefan on 2024/7/19.
//

#include "sense-voice-encoder.h"
#include <cmath>
#include <cassert>
#include <map>
#include <string>
#include <vector>
#define SENSE_VOICE_ENCODER_MAX_NODES 8192
#define WARP_SIZE 32
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
            /*.use_itn              =*/ false,
            /*.gpu_device           =*/ 0
    };
    return result;
}


static struct ggml_tensor *encoder_layer_sanm_forward(const sense_voice_hparams &hparams,
                                               sense_voice_context &sctx,
                                               ggml_context *ctx0,
                                               ggml_tensor *cur,
                                               sense_voice_layer_encoder &layer,
                                               ggml_cgraph *gf,
                                               bool user_flash_attn){

    const int n_state = hparams.n_encoder_hidden_state;
    const int n_head = hparams.n_encoder_attention_heads;
    auto state = sctx.state;

    struct ggml_tensor *residual = nullptr;

    if (layer.e_norm_w1->ne[0] == layer.e_norm_w2->ne[0]) {
        residual = ggml_cpy(
                ctx0, cur,
                ggml_new_tensor_3d(ctx0, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]));
    }

    {
        // layer norm
        // cur = ln_0_w*cur + ln_0_b
#ifdef GGML_CUDA
        int32_t dim_size = cur->ne[0];
        if (sctx.params.use_gpu && dim_size % WARP_SIZE) {
            int32_t pad_size = WARP_SIZE - (dim_size % WARP_SIZE);
            ggml_tensor *mean = ggml_mean(ctx0, cur);
            cur = ggml_sub(ctx0, cur, mean);
            ggml_tensor *sigma = ggml_mul(ctx0, cur, cur);
            sigma = ggml_sum_rows(ctx0, sigma);
            cur = ggml_scale(ctx0, ggml_div(ctx0, cur, ggml_sqrt(ctx0, sigma)), sqrt(dim_size));
            // cur = ggml_cont(ctx0, ggml_pad(ctx0, cur, pad_size, 0, 0, 0));
            // cur = ggml_norm(ctx0, cur, hparams.eps);
            // cur = ggml_cont(ctx0, ggml_view_4d(ctx0, cur, dim_size, cur->ne[1], cur->ne[2], cur->ne[3], cur->nb[1], cur->nb[2], cur->nb[3], 0));
            // cur = ggml_scale(ctx0, cur, sqrt(float(dim_size) / (dim_size + pad_size)));
        }else{
            cur = ggml_norm(ctx0, cur, hparams.eps);
        }
#else
        cur = ggml_norm(ctx0, cur, hparams.eps);
#endif
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w1), layer.e_norm_b1);
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
        int n_batch = cur->ne[2];
        Q = ggml_add(ctx0,
                     ggml_mul_mat_pad(ctx0, layer.e_attn_ln_q_w, cur),
                     layer.e_attn_ln_q_b);

        Q_h = ggml_reshape_4d(ctx0, Q, n_state / n_head, n_head, n_ctx, n_batch);
        Q_h = ggml_permute(ctx0, Q_h,  0, 2, 1, 3);
        Q_h = ggml_cont(ctx0, Q_h);

        ggml_set_name(Q_h, "attention_Q");

        K = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, layer.e_attn_ln_k_w, cur),
                     layer.e_attn_ln_k_b);

        K_h = ggml_reshape_4d(ctx0, K, n_state / n_head, n_head, n_ctx, n_batch);
        K_h = ggml_permute(ctx0, K_h, 0, 2, 1, 3);
        K_h = ggml_cont(ctx0, K_h);
        //      K = ggml_reshape_3d(ctx0, K, n_state, n_ctx, n_head);
        ggml_set_name(K_h, "attention_K");

        V = ggml_add(ctx0,
                     ggml_mul_mat(ctx0, layer.e_attn_ln_v_w, cur),
                     layer.e_attn_ln_v_b);
        ggml_set_name(V, "attention_V");

        V_h = ggml_reshape_4d(ctx0, V, n_state / n_head, n_head, n_ctx, n_batch);
        V_h = ggml_permute(ctx0, V_h, 0, 2, 1, 3);
        V_h = ggml_cont(ctx0, V_h);

        // fsmn forward with V
        int padding = (hparams.fsmn_kernel_size - 1) / 2;


        struct ggml_tensor *fsmn_memory = nullptr;
        // conv depth wise
        {
            {
                // implement conv depth wise with groups=input_channel implement
                // same in pytorch : F.conv1d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=n_state)
                struct ggml_tensor * a = layer.e_attn_fsmn_w;
                struct ggml_tensor * b = ggml_cont(ctx0, ggml_transpose(ctx0, V));
                // Process each batch separately and concatenate results
                // for (int i = 0; i < b->ne[2]; i++) {
                //     // View for current batch
                //     struct ggml_tensor *b_batch = ggml_view_3d(ctx0, b, b->ne[0], b->ne[1], 1, b->nb[1], b->nb[2], i * b->nb[2]);
                //     struct ggml_tensor *im2col = ggml_im2col(ctx0, a, ggml_reshape_4d(ctx0, b_batch, b_batch->ne[0], 1, b_batch->ne[1], b_batch->ne[2] * b_batch->ne[3]), 1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
                //     struct ggml_tensor * result = ggml_mul_mat(ctx0, a, im2col);
                //     struct ggml_tensor * fsmn_memory_batch = ggml_reshape_3d(ctx0, result, im2col->ne[1], b_batch->ne[1], b_batch->ne[2]);
                //     if (fsmn_memory == nullptr) {
                //         fsmn_memory = fsmn_memory_batch;
                //     } else {
                //         fsmn_memory = ggml_concat(ctx0, fsmn_memory, fsmn_memory_batch, 2);
                //     }
                // }
                struct ggml_tensor * im2col = ggml_im2col(ctx0, a, ggml_reshape_4d(ctx0, b, b->ne[0], 1, b->ne[1] * b->ne[2], b->ne[3]), 1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
                im2col = ggml_reshape_4d(ctx0, im2col, im2col->ne[0], im2col->ne[1], im2col->ne[2] / n_batch, n_batch);
                a = ggml_repeat(ctx0, ggml_cast(ctx0, a, GGML_TYPE_F32), ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, a->ne[0], a->ne[1], a->ne[2], n_batch));
                struct ggml_tensor * result = ggml_mul_mat(ctx0, a, im2col);
                fsmn_memory = ggml_reshape_3d(ctx0, result, im2col->ne[1], im2col->ne[2], im2col->ne[3]);
                // if(n_batch > 1){
                //     printf("n_batch: %d\n", n_batch);
                //     printf("a: %ld %ld %ld %ld\n", a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
                //     printf("b: %ld %ld %ld %ld\n", b->ne[0], b->ne[1], b->ne[2], b->ne[3]);
                //     printf("im2col: %ld %ld %ld %ld\n", im2col->ne[0], im2col->ne[1], im2col->ne[2], im2col->ne[3]);
                //     printf("result: %ld %ld %ld %ld\n", result->ne[0], result->ne[1], result->ne[2], result->ne[3]);
                //     printf("fsmn_memory: %ld %ld %ld %ld\n", fsmn_memory->ne[0], fsmn_memory->ne[1], fsmn_memory->ne[2], fsmn_memory->ne[3]);
                //     printf("V: %ld %ld %ld %ld\n", V->ne[0], V->ne[1], V->ne[2], V->ne[3]);
                // }
            }
            fsmn_memory = ggml_cont(ctx0, ggml_transpose(ctx0, fsmn_memory));
            fsmn_memory = ggml_add(ctx0, fsmn_memory, V);
            ggml_set_name(fsmn_memory, "fsmn_memory");
        }

        struct ggml_tensor *KQV;
        float KQscale = 1.0f / sqrtf(float(n_state) / n_head);

        if(user_flash_attn){
            const int n_ctx_pad = GGML_PAD(n_ctx, 256);
            const int n_state_head = n_state / n_head;

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, K, ggml_view_1d(ctx0, state->kv_pad.k, n_ctx*n_state*n_batch, 0)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, V, ggml_view_1d(ctx0, state->kv_pad.v, n_ctx*n_state*n_batch, 0)));

            struct ggml_tensor * K =
                    ggml_view_4d(ctx0, state->kv_pad.k,
                                 n_state_head, n_ctx_pad, n_head, n_batch,
                                 ggml_element_size(state->kv_pad.k)*n_state,
                                 ggml_element_size(state->kv_pad.k)*n_state_head,
                                 ggml_element_size(state->kv_pad.k)*n_state*n_ctx_pad,
                                 0);

            struct ggml_tensor * V =
                    ggml_view_4d(ctx0, state->kv_pad.v,
                                 n_state_head, n_ctx_pad, n_head, n_batch,
                                 ggml_element_size(state->kv_pad.v)*n_state,
                                 ggml_element_size(state->kv_pad.v)*n_state_head,
                                 ggml_element_size(state->kv_pad.v)*n_state*n_ctx_pad,
                                 0);
            KQV = ggml_flash_attn_ext(ctx0, Q_h, K, V, nullptr, KQscale, 0.0f, 0.0f);
            cur = ggml_reshape_3d(ctx0, KQV, n_state, n_ctx, n_batch);
        } else{
            // K * Q
            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K_h, Q_h);

            struct ggml_tensor *KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);


            KQV = ggml_mul_mat(
                    ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, V_h)), KQ_soft_max);
            struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_state, n_ctx, n_batch));
        }



        cur = ggml_cpy(ctx0, cur,
                       ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_state, n_ctx, n_batch));

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
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, cur->ne[0], cur->ne[1], cur->ne[2]));
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

    struct ggml_tensor *embedding = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 4, 1);
    ggml_set_name(embedding, "embedding");
    ggml_set_input(embedding);

    embedding = ggml_get_rows(ctx0, model->embedding, embedding);
    embedding = ggml_repeat(ctx0, embedding, ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, embedding->ne[0], embedding->ne[1], feature->ne[2]));

    struct ggml_tensor *cur = ggml_concat(ctx0, embedding, feature, 1);

    cur = ggml_scale(ctx0, cur, sqrtf(hparams.n_encoder_hidden_state));

    // implement encoder small forward graph
    //ref: https://github.com/modelscope/FunASR/blob/b7b4a83c18277a7022124cad790c08ae703b7a2d/funasr/models/sense_voice/model.py#L558-L583
    // [x] 1. sinusoidal position
    // [x] 2. encoders0
    // [x] 3. encoders
    // [x] 4. tp_encoders
    // [x] 5. tp_norm
    ggml_tensor *position = ggml_new_tensor_3d(ctx0, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    cur = ggml_add(ctx0, position, cur);

    // encoders0 forward
    cur = encoder_layer_sanm_forward(hparams, pctx, ctx0, cur, model->encoder->encoder0, gf, pctx.params.flash_attn);

    // encoders forward
    for (int i=0; i < hparams.n_encoder_layers - 1; i++){
        cur = encoder_layer_sanm_forward(hparams, pctx, ctx0, cur, model->encoder->encoders_layer[i], gf, pctx.params.flash_attn);
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
        cur = encoder_layer_sanm_forward(hparams,  pctx, ctx0, cur, model->encoder->tp_encoders_layer[i], gf, pctx.params.flash_attn);
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


    // encoder
    {
        auto & sched = state.sched_encode.sched;

        ggml_cgraph *gf = sense_voice_build_graph_encoder(ctx, state);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

//        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_get_tensor(gf, ));


        // set the inputs

        {
            struct ggml_tensor *position = ggml_graph_get_tensor(gf, "position");
            struct ggml_tensor *embedding = ggml_graph_get_tensor(gf, "embedding");


            auto n_len = position->ne[1];
            auto dim = position->ne[0];
            auto n_batch = position->ne[2];
            std::vector<float> _position;
            _position.resize(n_len * dim * n_batch);

            // construct position embedding
            // sinusoidal position embedding
            // reference:
            // https://github.com/modelscope/FunASR/blob/45d7aa9004763684fb748ee17942ecba81042201/funasr/models/transformer/embedding.py#L392-L405
            // P_{k,i} = sin(k/10000^(2i/d))  0 < i < d/2
            // p_{k,j} = cos(k/10000^(2j/d))  d/2 < j < d

            for (int b = 0; b < n_batch; b++)
            for (int k = 1; k <= n_len; k++) {
                for (int i = 0; i < dim / 2; i++) {
                _position[b * n_len * dim + (k - 1) * dim + i] = sinf(k * pow(10000, -2.0 * i / dim));
                _position[b * n_len * dim + (k - 1) * dim + i + dim / 2] =
                        cosf(k * pow(10000, -2.0 * i / dim));
                }
        	}


            ggml_backend_tensor_set(
                    position, _position.data(), 0,
                    ggml_nelements(position) * sizeof(float));

            int _embedding[4] = {ctx.language_id, 1, 2, ctx.params.use_itn ? 14 : 15};
            ggml_backend_tensor_set(embedding, _embedding, 0, 4*sizeof(int));
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
