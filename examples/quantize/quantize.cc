//
// Created by lovemefan on 2024/8/29.
//
#include "ggml.h"

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>



// quantize a model
static bool sense_voice_model_quantize(const std::string & fname_inp, const std::string & fname_out, ggml_ftype ftype) {

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    struct ggml_context * ctx = NULL;
    struct gguf_init_params gguf_params = {
            /*.no_alloc = */ false,
            /*.ctx      = */ &ctx,
    };

    struct gguf_context *gguf_ctx = gguf_init_from_file(fname_inp.c_str(), gguf_params);

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

    auto hparams = sense_voice_hparams{};

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

        // for the big tensors, we have the option to store the data in 16-bit
        // floats or quantized in order to save memory and also to speed up the
        // computation


        SENSE_VOICE_LOG_INFO("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        SENSE_VOICE_LOG_INFO("%s: n_encoder_hidden_state = %d\n", __func__,
                             hparams.n_encoder_hidden_state);
        SENSE_VOICE_LOG_INFO("%s: n_encoder_linear_units = %d\n", __func__,
                             hparams.n_encoder_linear_units);
        SENSE_VOICE_LOG_INFO("%s: n_encoder_attention_heads  = %d\n", __func__,
                             hparams.n_encoder_attention_heads);
        SENSE_VOICE_LOG_INFO("%s: n_encoder_layers = %d\n", __func__,
                             hparams.n_encoder_layers);

        SENSE_VOICE_LOG_INFO("%s: n_mels  = %d\n", __func__, hparams.n_mels);
        SENSE_VOICE_LOG_INFO("%s: ftype  = %d\n", __func__,
                             hparams.ftype);

    }
    auto vocab = sense_voice_vocab{};

    // load vocab
    {
        std::string word;

        const int token_idx = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
        const int n_vocab = gguf_get_arr_n(gguf_ctx, token_idx);

        if (n_vocab != hparams.n_vocab) {
            SENSE_VOICE_LOG_ERROR(
                    "%s: vocabulary loaded from model file error - vocabulary size is "
                    "%d, but got %d .\n",
                    __func__, hparams.n_vocab, n_vocab);
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

    vocab.n_vocab = hparams.n_vocab;

    // regexes of tensor names to not be quantized
    const std::vector<std::string> to_skip = {
            //"encoder.*",
            "embed.weight",
            "encoder.*.fsmn_block.weight",
            "encoder.encoders0.0.norm1.weight",
            "encoder.encoders0.0.self_attn.linear_q.weight",
            "encoder.encoders0.0.self_attn.linear_k.weight",
            "encoder.encoders0.0.self_attn.linear_v.weight",
            "ctc.ctc_lo.weight",
            // vad parameter
            "_model.stft.forward_basis_buffer.weight",
            "_model.encoder.*.reparam_conv.weight",
            "_model.encoder.*.reparam_conv.bias",
            "_model.decoder.rnn.weight_ih",
            "_model.decoder.rnn.weight_hh",
            "_model.decoder.decoder.2.weight",
            "_model.decoder.decoder.2.bias"
    };

    if (!sense_voice_ggml_quantize0(ctx, gguf_ctx, fname_inp, fname_out, ftype, 4, { ".*" }, to_skip)) {
        fprintf(stderr, "%s: failed to quantize model '%s'\n", __func__, fname_inp.c_str());
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n", argv[0]);
        ggml_print_ftypes(stderr);
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;


    {
        const int64_t t_start_us = ggml_time_us();

        if (!sense_voice_model_quantize(fname_inp, fname_out, ggml_ftype(ftype))) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}