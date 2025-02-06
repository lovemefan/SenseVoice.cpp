//
// Created by lovemefan on 2024/11/24.
//

#include "silero-vad.h"
#define SENSE_VOICE_VAD_MAX_NODES 1024
#define VAD_CHUNK_SIZE 640
/*
 \begin{array}{ll}
    i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
    o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
    c' = f * c + i * g \\
    h' = o * \tanh(c') \\
    \end{array}

 * */


ggml_cgraph *silero_vad_build_graph(
        sense_voice_context &ctx, sense_voice_state &state){

    const auto &model = ctx.vad_model.model;

    struct ggml_init_params params = {
            /*.mem_size   =*/state.sched_vad.meta.size(),
            /*.mem_buffer =*/state.sched_vad.meta.data(),
            /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, SENSE_VOICE_VAD_MAX_NODES, false);

    ggml_tensor *chunk = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, VAD_CHUNK_SIZE);
    // chunk size must be 576 before pad
    ggml_set_name(chunk, "audio_chunk");
    ggml_set_input(chunk);


    ggml_tensor *cur;
    // stft
    {
        cur = ggml_conv_1d(ctx0, model->stft.forward_basis_buffer, chunk, 128, 0, 1);
        // chunk operation by ggml view, equals torch.chunk(x, 2) in pytorch
        struct ggml_tensor * real_part = ggml_view_2d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->nb[1], 0);
        ggml_set_name(real_part, "real_part");
        struct ggml_tensor * image_part = ggml_view_2d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->nb[1], cur->nb[0] * cur->ne[0] * cur->ne[1] / 2);
        ggml_set_name(image_part, "image_part");
        // magnitude, equals torch.sqrt(real_part ** 2 + imag_part ** 2)
        cur = ggml_sqrt(ctx0,
                        ggml_add(ctx0,
                                 ggml_mul(ctx0, real_part, real_part),
                                 ggml_mul(ctx0, image_part, image_part)
                                 )
                        );
        ggml_set_name(cur, "magnitude");

    }

    // encoder
    {
        {
            cur = ggml_conv_1d(ctx0, model->encoders_layer[0].reparam_conv_w, cur, 1, 1, 1);
            cur = ggml_add(ctx0, cur, ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders_layer[0].reparam_conv_b)));
            cur = ggml_relu(ctx0, cur);

            cur = ggml_conv_1d(ctx0, model->encoders_layer[1].reparam_conv_w, cur, 2, 1, 1);
            cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders_layer[1].reparam_conv_b)));
            cur = ggml_relu(ctx0, cur);

            cur = ggml_conv_1d(ctx0, model->encoders_layer[2].reparam_conv_w, cur, 2, 1, 1);
            cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders_layer[2].reparam_conv_b)));
            cur = ggml_relu(ctx0, cur);

            cur = ggml_conv_1d(ctx0, model->encoders_layer[3].reparam_conv_w, cur, 1, 1, 1);
            cur = ggml_add(ctx0, cur,  ggml_cont(ctx0, ggml_transpose(ctx0, model->encoders_layer[3].reparam_conv_b)));
            cur = ggml_relu(ctx0, cur);
        }

    }

    //decoder
    {

        struct ggml_tensor* in_lstm_hidden_state = ggml_new_tensor_1d(ctx0, cur->type, cur->ne[1]);
        struct ggml_tensor*  in_lstm_context = ggml_new_tensor_1d(ctx0, cur->type, cur->ne[1]);

        struct ggml_tensor* out_lstm_hidden_state;
        struct ggml_tensor*  out_lstm_context;

        ggml_set_name(in_lstm_context, "in_lstm_context");
        ggml_set_name(in_lstm_hidden_state, "in_lstm_hidden_state");


        // lstm cell
        // ref: https://github.com/pytorch/pytorch/blob/1a93b96815b5c87c92e060a6dca51be93d712d09/aten/src/ATen/native/RNN.cpp#L298-L304
        // gates = x @ self.weight_ih.T + self.bias_ih + hx[0] @ self.weight_hh.T + self.bias_hh
        // chunked_gates = gates.chunk(4, dim=-1)
        // ingate = torch.sigmoid(chunked_gates[0])
        // forgetgate = torch.sigmoid(chunked_gates[1])
        // cellgate = torch.tanh(chunked_gates[2])
        // outgate = torch.sigmoid(chunked_gates[3])
        // cy = forgetgate * hx[1] + ingate * cellgate
        // hy = outgate * torch.tanh(cy)

        struct ggml_tensor *gates = ggml_add(
                ctx0,
                ggml_add(ctx0, ggml_mul_mat(ctx0,
                                            model->decoder.lstm_weight_ih,
                                            ggml_transpose(ctx0, cur)),
                         model->decoder.lstm_bias_ih),

                ggml_add(ctx0, ggml_mul_mat(ctx0,
                                            model->decoder.lstm_weight_hh,
                                            in_lstm_hidden_state),
                         model->decoder.lstm_bias_hh));
        ggml_set_name(gates, "gates");

        struct ggml_tensor * input_gates = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, gates, gates->ne[0] / 4, gates->ne[1] , gates->nb[1], 0));
        struct ggml_tensor * forget_gates = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], gates->nb[0] / 4 * gates->ne[0]));
        struct ggml_tensor * cell_gate = ggml_tanh(ctx0, ggml_view_2d(ctx0, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], 2 * gates->nb[0] / 4 * gates->ne[0]));
        struct ggml_tensor * out_gates = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, gates, gates->ne[0] / 4, gates->ne[1], gates->nb[1], 3 * gates->nb[0] / 4 * gates->ne[0]));

        ggml_set_name(input_gates, "input_gates");
        ggml_set_name(forget_gates, "forget_gates");
        ggml_set_name(cell_gate, "cell_gates");
        ggml_set_name(out_gates, "out_gates");

        out_lstm_context = ggml_add(ctx0,
                                          ggml_mul(ctx0, forget_gates, in_lstm_context),
                                          ggml_mul(ctx0, input_gates, cell_gate)
                                          );

        ggml_set_name(out_lstm_context, "out_lstm_context");
        ggml_set_output(out_lstm_context);
        out_lstm_hidden_state = ggml_mul(ctx0, out_gates, ggml_tanh(ctx0, out_lstm_context));
        ggml_set_name(out_lstm_hidden_state, "out_lstm_hidden_state");
        ggml_set_output(out_lstm_hidden_state);

        cur = ggml_relu(ctx0, out_lstm_hidden_state);
        cur = ggml_conv_1d(ctx0, model->decoder.decoder_conv_w, ggml_cont(ctx0, ggml_transpose(ctx0, cur)), 1, 0, 1);
        cur = ggml_add(ctx0, cur, ggml_transpose(ctx0, model->decoder.decoder_conv_b));
        ggml_set_name(cur, "decoder_out");
        cur = ggml_sigmoid(ctx0, cur);
        ggml_set_name(cur, "logit");

    }

    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}


bool silero_vad_encode_internal(sense_voice_context &ctx,
                                sense_voice_state &state,
                                std::vector<float> chunk,
                                const int n_threads,
                                float &speech_prob){
    {
        auto & sched = ctx.state->sched_vad.sched;
        ggml_cgraph *gf = silero_vad_build_graph(ctx, state);

        //          ggml_backend_sched_set_eval_callback(sched,  ctx->params.cb_eval, &ctx->params.cb_eval_user_data);


        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }
        // set the input
        {

            struct ggml_tensor *data = ggml_graph_get_tensor(gf, "audio_chunk");
            ggml_backend_tensor_set(data, chunk.data(), 0, ggml_nbytes(data));

            struct ggml_tensor *in_lstm_context = ggml_graph_get_tensor(gf, "in_lstm_context");
            struct ggml_tensor *in_lstm_hidden_state = ggml_graph_get_tensor(gf, "in_lstm_hidden_state");

            ggml_backend_tensor_copy(state.vad_lstm_context, in_lstm_context);
            ggml_backend_tensor_copy(state.vad_lstm_hidden_state, in_lstm_hidden_state);

        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }

        // save output state
        {
            struct ggml_tensor *lstm_context = ggml_graph_get_tensor(gf, "out_lstm_context");
            ggml_backend_tensor_copy(lstm_context, state.vad_lstm_context);
            struct ggml_tensor *lstm_hidden_state = ggml_graph_get_tensor(gf, "out_lstm_hidden_state");
            ggml_backend_tensor_copy(lstm_hidden_state, state.vad_lstm_hidden_state);

        }
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logit"), &speech_prob, 0, sizeof(speech_prob));
    }
    return true;
}
