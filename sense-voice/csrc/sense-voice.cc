//
// Created by lovemefan on 2024/7/19.
//
#include "sense-voice.h"
#include "sense-voice-encoder.h"
#include "common.h"
#include <ggml.h>
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
#define SENSE_VOICE_MAX_NODES 8192

int sense_voice_lang_id(const char * lang) {
    if (!g_lang.count(lang)) {
        for (const auto & kv : g_lang) {
            if (kv.second.second == lang) {
                return kv.second.first;
            }
        }

        SENSE_VOICE_LOG_ERROR("%s: unknown language '%s'\n", __func__, lang);
        return -1;
    }
    return g_lang.at(lang).first;
}


// load the model from a gguf file
// see the convert-pt-to-ggml.py script for details
bool sense_voice_model_load(const char *path_model, sense_voice_context &sctx) {
    struct ggml_context *ctx_data = NULL;
    
    struct gguf_init_params gguf_params = {
            /*.no_alloc = */ false,
            /*.ctx      = */ &ctx_data,
    };
    struct gguf_context *gguf_ctx = gguf_init_from_file(path_model, gguf_params);

    // kv data
    {
        SENSE_VOICE_LOG_INFO("%s: version:      %d\n", __func__,
                            gguf_get_version(gguf_ctx));
        SENSE_VOICE_LOG_INFO("%s: alignment:   %zu\n", __func__,
                            gguf_get_alignment(gguf_ctx));
        SENSE_VOICE_LOG_INFO("%s: data offset: %zu\n", __func__,
                            gguf_get_data_offset(gguf_ctx));

        const int n_kv = gguf_get_n_kv(gguf_ctx);

        SENSE_VOICE_LOG_DEBUG("%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char *key = gguf_get_key(gguf_ctx, i);
            SENSE_VOICE_LOG_DEBUG("%s: %s        \n", __func__, key);
        }
    }

    SENSE_VOICE_LOG_INFO("%s: loading model\n", __func__);

    const int64_t t_start_ms = ggml_time_ms();

    sctx.t_start_ms = t_start_ms;

    auto &sense_voice = sctx.model;
    auto &vocab = sctx.vocab;
    auto &hparams = sense_voice.hparams;
    sense_voice.model_type =  gguf_get_val_str(gguf_ctx, 0);
    // load hparams
    {

        hparams.n_vocab = gguf_get_val_i32(
                gguf_ctx, gguf_find_key(gguf_ctx, "tokenizer.vocab_size"));
        hparams.n_encoder_hidden_state =
                gguf_get_val_i32(gguf_ctx, gguf_find_key(gguf_ctx, "encoder.output_size"));
        hparams.n_encoder_linear_units = gguf_get_val_i32(
                gguf_ctx, gguf_find_key(gguf_ctx, "encoder.linear_units"));
        hparams.n_encoder_attention_heads = gguf_get_val_i32(
                gguf_ctx, gguf_find_key(gguf_ctx, "encoder.attention_heads"));
        hparams.n_encoder_layers = gguf_get_val_i32(
                gguf_ctx, gguf_find_key(gguf_ctx, "encoder.num_blocks"));
        hparams.n_tp_encoder_layers = gguf_get_val_i32(
                gguf_ctx, gguf_find_key(gguf_ctx, "encoder.tp_blocks"));

        if (sense_voice.model_type == "SenseVoiceLarge") {
            hparams.n_decoder_hidden_state =
                    gguf_get_val_i32(gguf_ctx, gguf_find_key(gguf_ctx, "model.inner_dim"));
            hparams.n_decoder_linear_units = gguf_get_val_i32(
                    gguf_ctx, gguf_find_key(gguf_ctx, "decoder.linear_units"));
            hparams.n_decoder_attention_heads = gguf_get_val_i32(
                    gguf_ctx, gguf_find_key(gguf_ctx, "decoder.attention_heads"));
            hparams.n_decoder_layers = gguf_get_val_i32(
                    gguf_ctx, gguf_find_key(gguf_ctx, "decoder.num_blocks"));
        }

        // for the big tensors, we have the option to store the data in 16-bit
        // floats or quantized in order to save memory and also to speed up the
        // computation
        sctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype)(sense_voice.hparams.ftype));
        if (sctx.wtype == GGML_TYPE_COUNT) {
            SENSE_VOICE_LOG_INFO("%s: invalid model (bad ftype value %d)\n", __func__,
                                 sense_voice.hparams.ftype);
            return false;
        }

        const size_t scale = sense_voice.hparams.ftype ? 1 : 2;

        SENSE_VOICE_LOG_INFO("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        SENSE_VOICE_LOG_INFO("%s: n_encoder_hidden_state = %d\n", __func__,
                            hparams.n_encoder_hidden_state);
        SENSE_VOICE_LOG_INFO("%s: n_encoder_linear_units = %d\n", __func__,
                            hparams.n_encoder_linear_units);
        SENSE_VOICE_LOG_INFO("%s: n_encoder_attention_heads  = %d\n", __func__,
                            hparams.n_encoder_attention_heads);
        SENSE_VOICE_LOG_INFO("%s: n_encoder_layers = %d\n", __func__,
                            hparams.n_encoder_layers);

        if (sense_voice.model_type == "SenseVoiceLarge") {
            SENSE_VOICE_LOG_INFO("%s: n_decoder_hidden_state  = %d\n", __func__,
                                 hparams.n_decoder_hidden_state);
            SENSE_VOICE_LOG_INFO("%s: n_decoder_linear_units  = %d\n", __func__,
                                 hparams.n_decoder_linear_units);
            SENSE_VOICE_LOG_INFO("%s: n_decoder_attention_heads   = %d\n", __func__,
                                 hparams.n_decoder_attention_heads);
            SENSE_VOICE_LOG_INFO("%s: n_decoder_layers  = %d\n", __func__,
                                 hparams.n_decoder_layers);
        }
        SENSE_VOICE_LOG_INFO("%s: n_mels  = %d\n", __func__, hparams.n_mels);
        SENSE_VOICE_LOG_INFO("%s: ftype  = %d\n", __func__,
                             sense_voice.hparams.ftype);


        // initialize all memory buffers
        // always have at least one decoder

        sctx.model.buf = new std::vector<uint8_t>();
//        sctx.model.buf->resize(scale * MEM_REQ_MODEL.at(sctx.wtype).at(model.type));

        // we skip initialization of the state until it is needed
        // because it might be that state will always be provided externally.
    }

    // load vocab
    {
        std::string word;

        const int token_idx = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
        const int n_vocab = gguf_get_arr_n(gguf_ctx, token_idx);

        if (n_vocab != sense_voice.hparams.n_vocab) {
            SENSE_VOICE_LOG_ERROR(
                    "%s: vocabulary loaded from model file error - vaocabulary size is "
                    "%d, but got %d .\n",
                    __func__, sense_voice.hparams.n_vocab, n_vocab);
        }
        std::vector<const char *> tokens;
        tokens.resize(n_vocab);
        for (uint32_t i = 0; i < n_vocab; i++) {
            word = gguf_get_arr_str(gguf_ctx, token_idx, i);
            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }

        SENSE_VOICE_LOG_INFO("%s: vocab[%d] loaded\n", __func__, n_vocab);
    }

    vocab.n_vocab = sense_voice.hparams.n_vocab;

    size_t ctx_size = 0;

    const ggml_type wtype = sctx.wtype;
    const ggml_type vtype =
            sctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;  // conv

    {
        const auto &hparams = sense_voice.hparams;



        // load weights
        {
            const int n_tensors = gguf_get_n_tensors(gguf_ctx);
            SENSE_VOICE_LOG_INFO("%s: n_tensors: %d\n", __func__, n_tensors);
            sense_voice.n_loaded = 0;

            for (int i = 0; i < n_tensors; ++i) {
                const std::string name = gguf_get_tensor_name(gguf_ctx, i);
                struct ggml_tensor *cur = ggml_get_tensor(ctx_data, name.c_str());
                sense_voice.tensors[name] = cur;

                auto n_dim = ggml_n_dims(cur);
                std::stringstream shape;
                if (n_dim == 1)
                    shape << cur->ne[0];
                else if (n_dim == 2)
                    shape << cur->ne[0] << ',' << cur->ne[1];
                else if (n_dim == 3)
                    shape << cur->ne[0] << ',' << cur->ne[1] << ',' << cur->ne[2];
                else
                    shape << cur->ne[0] << ',' << cur->ne[1] << ',' << cur->ne[2] << ','
                          << cur->ne[3];

                SENSE_VOICE_LOG_DEBUG(
                        "%s: tensor[%d]: n_dims = %d, shape = (%s), name = %s, "
                        "data = %p\n",
                        __func__, i, ggml_n_dims(cur), shape.str().c_str(), cur->name,
                        cur->data);
            }
        }
    }

    {
        // build model
        sense_voice.model = new struct sense_voice;
        sense_voice.model->encoder = new struct sense_voice_encoder;
        sense_voice.model->encoder->encoders_layer =  std::vector<sense_voice_layer_encoder>(hparams.n_encoder_layers - 1);
        sense_voice.model->encoder->tp_encoders_layer =  std::vector<sense_voice_layer_encoder>(hparams.n_tp_encoder_layers);



        // load encoder weights, multi layers of EncoderLayerSANM
        {
            sense_voice.model->embedding = sense_voice.tensors["embed.weight"];
            std::vector<sense_voice_layer_encoder> tmp_encoder0;
            tmp_encoder0.push_back(sense_voice.model->encoder->encoder0);
            set_sense_voice_encoder_layer_sanm(tmp_encoder0, sense_voice.tensors, 1, "encoders0");
            sense_voice.model->encoder->encoder0 = tmp_encoder0[0];
            set_sense_voice_encoder_layer_sanm(sense_voice.model->encoder->encoders_layer, sense_voice.tensors, hparams.n_encoder_layers - 1, "encoders");
            set_sense_voice_encoder_layer_sanm(sense_voice.model->encoder->tp_encoders_layer, sense_voice.tensors, hparams.n_tp_encoder_layers, "tp_encoders");

            sense_voice.model->encoder->e_after_norm_w = sense_voice.tensors["encoder->after_norm.weight"];
            sense_voice.model->encoder->e_after_norm_b = sense_voice.tensors["encoder->after_norm.bias"];

            sense_voice.model->encoder->e_after_norm_w = sense_voice.tensors["encoder->tp_norm.weight"];
            sense_voice.model->encoder->e_after_norm_b = sense_voice.tensors["encoder.tp_norm.bias"];

            sense_voice.model->ctc_out_linear_weight = sense_voice.tensors["ctc.ctc_lo.weight"];
            sense_voice.model->ctc_out_linear_bias = sense_voice.tensors["ctc.ctc_lo.bias"];

        }
    }


    // decoder layers
    if(sense_voice.model_type == "SenseVoiceLarge")
    {

    }

    sctx.t_load_ms = ggml_time_ms() - t_start_ms;
    SENSE_VOICE_LOG_INFO("%s: load %s takes %f second \n", __func__, sense_voice.model_type.c_str(),
                        sctx.t_load_ms * 1.0 / 1000);
    return true;
}

struct sense_voice_context *sense_voice_init_with_params_no_state(
        const char *path_model, sense_voice_context_params params) {
    ggml_time_init();

    auto *ctx = new struct sense_voice_context;

    ctx->params = params;

    if (!sense_voice_model_load(path_model, *ctx)) {
        SENSE_VOICE_LOG_ERROR("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

struct sense_voice_context * sense_voice_small_init_from_file_with_params_no_state(const char * path_model, struct sense_voice_context_params params) {
    SENSE_VOICE_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);
    

    auto ctx = sense_voice_init_with_params_no_state(path_model, params);

    if (ctx) {
        ctx->path_model = path_model;
    }

    return ctx;
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
        SENSE_VOICE_LOG_INFO("%s: using Metal backend\n", __func__);
        ggml_backend_metal_log_set_callback(g_state.log_callback,
                                            g_state.log_callback_user_data);
        backend_gpu = ggml_backend_metal_init();
        if (!backend_gpu) {
            SENSE_VOICE_LOG_ERROR("%s: ggml_backend_metal_init() failed\n", __func__);
        } else if (!ggml_backend_metal_supports_family(backend_gpu, 7)) {
            SENSE_VOICE_LOG_ERROR(
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

void sense_voice_free_state(struct sense_voice_state *state) {
    if (state) {
#ifdef WHISPER_USE_COREML
        if (state->ctx_coreml != nullptr) {
            whisper_coreml_free(state->ctx_coreml);
            state->ctx_coreml = nullptr;
        }
#endif


        ggml_gallocr_free(state->alloc_encode.alloc);
        ggml_gallocr_free(state->alloc_decode.alloc);

        ggml_backend_free(state->backend);

        delete state;
    }
}

// measure the memory usage of a graph and prepare the allocr's internal data
// buffer
static bool sense_voice_allocr_graph_init(
        struct sense_voice_allocr &allocr, ggml_backend_t backend,
        std::function<struct ggml_cgraph *()> &&get_graph) {
    auto &alloc = allocr.alloc;
    auto &meta = allocr.meta;

    alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    meta.resize(ggml_tensor_overhead() * SENSE_VOICE_MAX_NODES +
                ggml_graph_overhead());

    // since there are dependencies between the different graphs,
    // we need to allocate them instead of only reserving to get the correct
    // compute buffer size
    if (!ggml_gallocr_alloc_graph(alloc, get_graph())) {
        // failed to allocate the compute buffer
        SENSE_VOICE_LOG_ERROR("%s: failed to allocate the compute buffer\n",
                             __func__);
        return false;
    }
    return true;
}

static size_t sense_voice_allocr_size(struct sense_voice_allocr &allocr) {
    return allocr.meta.size() + ggml_gallocr_get_buffer_size(allocr.alloc, 0);
}

struct sense_voice_state *sense_voice_init_state(sense_voice_context *ctx) {
    ctx->state = new sense_voice_state;
    auto state = ctx->state;
    
    state->backend = sense_voice_backend_init(ctx->params);
    if (!state->backend) {
        SENSE_VOICE_LOG_ERROR("%s: sense_voice_backend_init() failed\n", __func__);
        sense_voice_free_state(state);
        return nullptr;
    }

#ifdef USE_COREML
    const auto path_coreml = PARAFORMER_get_coreml_path_encoder(ctx->path_model);

    SENSE_VOICE_LOG_INFO("%s: loading Core ML model from '%s'\n", __func__,
                        path_coreml.c_str());
    SENSE_VOICE_LOG_INFO("%s: first run on a device may take a while ...\n",
                        __func__);

    state->ctx_coreml = PARAFORMER_coreml_init(path_coreml.c_str());
    if (!state->ctx_coreml) {
        SENSE_VOICE_LOG_INFO("%s: failed to load Core ML model from '%s'\n",
                            __func__, path_coreml.c_str());
#ifndef COREML_ALLOW_FALLBACK
        delete state;
        return nullptr;
#endif
    } else {
        SENSE_VOICE_LOG_INFO("%s: Core ML model loaded\n", __func__);
    }
#endif

    state->logits_id.reserve(ctx->model.hparams.n_vocab);

    // encoder allocator
    {
        bool ok = sense_voice_allocr_graph_init(
                state->alloc_encode, state->backend,
                [&]() { return sense_voice_build_graph_encoder(*ctx, *state); });

        if (!ok) {
            SENSE_VOICE_LOG_ERROR("%s: failed to init encode allocator\n", __func__);
            sense_voice_free_state(state);
            return nullptr;
        }

        SENSE_VOICE_LOG_INFO("%s: compute buffer (all)   = %7.2f MB\n", __func__,
                            sense_voice_allocr_size(state->alloc_encode) / 1e6);
    }


    // todo decoder allocator
    {
        //    SENSE_VOICE_LOG_INFO(
        //        "%s: compute buffer (decode) = %7.2f MB\n", __func__,
        //        paraformer_allocr_size(state->alloc_decode) / 1024.0 / 1024.0);
    }

    return state;
}

struct sense_voice_context * sense_voice_small_init_from_file_with_params(const char * path_model, struct sense_voice_context_params params) {
    sense_voice_context * ctx = sense_voice_small_init_from_file_with_params_no_state(path_model, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = sense_voice_init_state(ctx);
//    if (!ctx->state) {
//        sense_voice_small_free(ctx);
//        return nullptr;
//    }

    return ctx;
}