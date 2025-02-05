//
// Created by lovemefan on 2024/7/19.
//

#ifndef SENSEVOICE_CPP_COMMON_H
#define SENSEVOICE_CPP_COMMON_H

#include <cstdint>
#include <string>
#include <map>
#include <set>
#include <ggml-cpu.h>
#include <ggml-cpp.h>
#include <gguf.h>
#include "sense-voice-frontend.h"


#ifdef __GNUC__
#define SENSEVOICE_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#define SENSEVOICE_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#define SENSEVOICE_DEPRECATED(func, hint) func
#endif

#ifdef SENSEVOICE_SHARED
#ifdef _WIN32
#ifdef SENSEVOICE_BUILD
#define SENSEVOICE_API __declspec(dllexport)
#else
#define SENSEVOICE_API __declspec(dllimport)
#endif
#else
#define SENSEVOICE_API __attribute__((visibility("default")))
#endif
#else
#define SENSEVOICE_API
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

#if defined(GGML_BIG_ENDIAN)
#include <bit>
#include <math.h>
#include "ggml-alloc.h"
#include "ggml-backend.h"

template <typename T>
static T byteswap(T value) {
    return std::byteswap(value);
}

template <>
float byteswap(float value) {
    return std::bit_cast<float>(byteswap(std::bit_cast<std::uint32_t>(value)));
}

template <typename T>
static void byteswap_tensor_data(ggml_tensor *tensor) {
    T *datum = reinterpret_cast<T *>(tensor->data);
    for (int i = 0; i < ggml_nelements(tensor); i++) {
        datum[i] = byteswap(datum[i]);
    }
}

static void byteswap_tensor(ggml_tensor *tensor) {
    switch (tensor->type) {
        case GGML_TYPE_I16: {
            byteswap_tensor_data<int16_t>(tensor);
            break;
        }
        case GGML_TYPE_F16: {
            byteswap_tensor_data<ggml_fp16_t>(tensor);
            break;
        }
        case GGML_TYPE_I32: {
            byteswap_tensor_data<int32_t>(tensor);
            break;
        }
        case GGML_TYPE_F32: {
            byteswap_tensor_data<float>(tensor);
            break;
        }
        default: {  // GML_TYPE_I8
            break;
        }
    }
}

#define BYTESWAP_VALUE(d) d = byteswap(d)
#define BYTESWAP_FILTERS(f)      \
  do {                           \
    for (auto &datum : f.data) { \
      datum = byteswap(datum);   \
    }                            \
  } while (0)
#define BYTESWAP_TENSOR(t) \
  do {                     \
    byteswap_tensor(t);    \
  } while (0)
#else
#define BYTESWAP_VALUE(d) \
  do {                    \
  } while (0)
#define BYTESWAP_FILTERS(f) \
  do {                      \
  } while (0)
#define BYTESWAP_TENSOR(t) \
  do {                     \
  } while (0)
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define SENSE_VOICE_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define SENSE_VOICE_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define SENSE_VOICE_ATTRIBUTE_FORMAT(...)
#endif

#define SENSEVOICE_LOG_ERROR(...) \
  sense_voice_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define SENSEVOICE_LOG_WARN(...) \
  sense_voice_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define SENSEVOICE_LOG_INFO(...) \
  sense_voice_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)

// define this to enable verbose trace logging - useful for debugging purposes
#define SENSEVOICE_DEBUG

#if defined(SENSEVOICE_DEBUG)
#define SENSEVOICE_LOG_DEBUG(...) \
  sense_voice_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define SENSEVOICE_LOG_DEBUG(...)
#endif

#define SENSEVOICE_ASSERT(x)                                           \
  do {                                                                 \
    if (!(x)) {                                                        \
      SENSEVOICE_LOG_ERROR("SENSEVOICE_ASSERT: %s:%d: %s\n", __FILE__, \
                           __LINE__, #x);                              \
      abort();                                                         \
    }                                                                  \
  } while (0)
//
// logging
//

SENSE_VOICE_ATTRIBUTE_FORMAT(2, 3)
void sense_voice_log_internal        (ggml_log_level level, const char * format, ...);
void sense_voice_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define SENSE_VOICE_LOG_ERROR(...) sense_voice_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define SENSE_VOICE_LOG_WARN(...)  sense_voice_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define SENSE_VOICE_LOG_INFO(...)  sense_voice_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)

// define this to enable verbose trace logging - useful for debugging purposes
//#define SENSE_VOICE_DEBUG

#if defined(SENSE_VOICE_DEBUG)
#define SENSE_VOICE_LOG_DEBUG(...) sense_voice_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define SENSE_VOICE_LOG_DEBUG(...)
#endif

#define SENSE_VOICE_ASSERT(x) \
    do { \
        if (!(x)) { \
            SENSE_VOICE_LOG_ERROR("SENSE_VOICE_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

struct sense_voice_global {
    // We save the log callback globally
    ggml_log_callback log_callback = sense_voice_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static sense_voice_global g_state;

// available whisper models
enum e_model {
    MODEL_SMALL,
    MODEL_LARGE,
};
struct sense_voice_encoder;

struct sense_voice {
    struct ggml_tensor *embedding;

    struct sense_voice_encoder *encoder;

    // ctc  out
    struct ggml_tensor *ctc_out_linear_weight;
    struct ggml_tensor *ctc_out_linear_bias;

};

struct silero_vad;

struct silero_vad_model {
    struct silero_vad *model;
    // context
    struct ggml_context *ctx;

    // the model backend data is read-only and can be shared between processors
    ggml_backend_buffer_t buffer = nullptr;
};


static const std::map<std::string, std::pair<int, std::string>> g_lang = {
        { "auto",  { 0,  "auto",         } },
        { "zh",  { 3,  "chinese",         } },
        { "en",  { 4,  "english",          } },
        { "yue",  { 7,  "cantonese",         } },
        { "ja",  { 11,  "japanese",         } },
        { "ko",  { 12,  "korean",          } },
        { "nospeech",  { 13,  "nospeech",          } },
};

struct sense_voice_hparams {
    int n_vocab = 25055;                // number of vocab
    int n_max_audio_length = 20000;    //
    int n_encoder_hidden_state = 512;  // dim of hidden state
    int n_encoder_linear_units = 2048;
    int n_encoder_attention_heads = 4;  // head of self attention
    int n_encoder_layers = 50;          // num block of encoder
    int n_tp_encoder_layers = 20;
    int n_encoder_0_norm_size = 560;
    int n_decoder_hidden_state = 512;
    int n_decoder_linear_units = 2048;
    int n_decoder_attention_heads = 4;
    int n_decoder_layers = 14;
    int fsmn_kernel_size = 11;
    int n_vad_encoder_layers = 4;
    int n_predictor_dim = 512;
    float predictor_tail_threshold = 0.45;

    // for auto-detection, set to nullptr, "" or "auto"
    const char * language;

    int n_mels = 80;  // dim of mels
    std::string window = "hamming";
    int frame_length = 25;
    int frame_shift = 10;
    int lfr_m = 7;
    int lfr_n = 6;
    int ftype = 1;
    float eps = 1e-5f;
    int n_audio_ctx = 1600;
};

// replace std::pair by using customized pair struct (reason: std::pair is very
// slow)
// template <typename A, typename B>
// struct paraformer_pair {
//     A first;
//     B second;

//     // Define a constructor that takes two arguments.
//     paraformer_pair(const A &a, const B &b) : first(a), second(b) {}
//     // Define a constructor that takes no argument.
//     paraformer_pair() : first(A()), second(B()) {}
// };


// ggml_allocr wrapper for usage
// struct sense_voice_allocr {
//     ggml_gallocr_t alloc = nullptr;
//     std::vector<uint8_t> meta;
// };

// struct sense_voice_token_data {
//     int id;   // token id
//     int tid;  // forced timestamp token id

//     float p;      // probability of the token
//     float plog;   // log probability of the token
//     float pt;     // probability of the timestamp token
//     float ptsum;  // sum of probabilities of all timestamp tokens

//     // token-level timestamp data
//     // do not use if you haven't computed token-level timestamps
//     int64_t t0;  // start time of the token
//     int64_t t1;  //   end time of the token

//     float vlen;  // voice length of the token
// };

struct sense_voice_segment {
    size_t t0;                   // 时间区间左端点
    size_t t1;                   // 时间区间右端点
    // std::string text;         // tokens对应的文本
    std::vector<int> tokens;     // 识别后的tokens
    std::vector<double> samples;  // 具体音频
    // std::vector<float> 
    // bool speaker_turn_next;
};

struct sense_voice_vocab {
    using id = int32_t;
    using token = std::string;

    int n_vocab = 25055;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    id token_eot = 2;
    id token_sot = 1;
};

// ggml_backend_sched wrapper for sense_voice usage
struct sense_voice_sched {
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
};

struct sense_voice_kv_cell {
    int32_t pos = -1;

    std::set<int32_t> seq_id;

    bool has_seq_id(const int32_t & id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

struct sense_voice_kv_cache {
    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<sense_voice_kv_cell> cells;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx = nullptr;

    ggml_backend_buffer_t buffer = nullptr;
};

struct sense_voice_state {
    // int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    // int64_t t_prompt_us = 0;
    int64_t t_feature_us = 0;
    // int32_t n_sample = 0;  // number of tokens sampled
    // int32_t n_encode = 0;  // number of encoder calls
    // int32_t n_decode =
    //         0;  // number of decoder calls with n_tokens == 1 (text-generation)
    // int32_t n_prompt =
    //         0;  // number of decoder calls with n_tokens >  1 (prompt encoding)
    // int32_t n_fail_p = 0;  // number of logprob threshold failures
    // int32_t n_fail_h = 0;  // number of entropy threshold failures

    float duration = 0;
    // padded buffer for flash-attention
    sense_voice_kv_cache kv_pad;

    // shared between all decoders
    sense_voice_feature feature;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    // std::vector<uint8_t> work_buffer;

    std::vector<ggml_backend_t> backends;

    sense_voice_sched sched_vad;
    sense_voice_sched sched_vad_sate;
    sense_voice_sched sched_encode;
    sense_voice_sched sched_decode;

    // hidden state in vad lstm
    ggml_context * vad_ctx = nullptr;
    struct ggml_tensor *  vad_lstm_hidden_state;
    struct ggml_tensor * vad_lstm_context;
    ggml_backend_buffer_t vad_lstm_hidden_state_buffer = nullptr;
    ggml_backend_buffer_t vad_lstm_context_buffer = nullptr;

    ggml_cgraph *sense_voice_encoder_graph;
    ggml_cgraph *sense_voice_decoder_graph;

    // result of the encoder
    struct ggml_tensor *encoder_out = nullptr;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<int> ids;
    std::vector<sense_voice_segment> result_all;
    std::vector<size_t> segmentIDs;
    // std::vector<int> prompt_past;

    // work container used to avoid memory allocations
    // std::vector<paraformer_pair<double, sense_voice_vocab::id>> logits_id;

    int lang_id = 0;  // english by default

    std::string path_model;  // populated by PARAFORMER_init_from_file()
#ifdef USE_COREML
    sense_voice_coreml_context *ctx_coreml = nullptr;
#endif


    // [EXPERIMENTAL] token-level timestamps data
    // int64_t t_beg = 0;
    // int64_t t_last = 0;
    // int tid_last;
    // std::vector<float> energy;  // PCM signal energy

    // [EXPERIMENTAL] speed-up techniques
    int32_t exp_n_audio_ctx = 0;  // 0 - use default
};

// Progress callback
typedef void (*sense_voice_progress_callback)(struct sense_voice_context *ctx,
                                             struct sense_voice_state *state,
                                             int progress, void *user_data);

// Available sampling strategies
enum sense_voice_decoding_strategy {
    SENSE_VOICE_SAMPLING_GREEDY,
    SENSE_VOICE_SAMPLING_BEAM_SEARCH,
};

struct sense_voice_full_params {
    enum sense_voice_decoding_strategy strategy;
    int n_threads;
    const char *language;
    int n_max_text_ctx;  // max tokens to use from past text as prompt for the
                          // decoder
    int offset_ms;       // start offset in ms
    int duration_ms;     // audio duration to process in ms

    bool no_timestamps;     // do not generate timestamps
    bool single_segment;    // force single segment output (useful for streaming)
    bool print_progress;    // print progress information
    bool print_timestamps;  // print timestamps for each text segment when
                          // printing realtime

    bool debug_mode;  // enable debug_mode provides extra info (eg. Dump log_mel)
    int  audio_ctx;

    struct {
        int best_of;
    } greedy;

    struct {
        int beam_size;
    } beam_search;

    // called on each progress update
    sense_voice_progress_callback progress_callback;
    void *progress_callback_user_data;

};



struct sense_voice_model {
    std::string model_type;
    sense_voice_hparams hparams;
    silero_vad *vad_model;
    sense_voice *model;
    // context
    struct ggml_context *ctx;

    // the model backend data is read-only and can be shared between processors
    ggml_backend_buffer_t buffer = nullptr;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct sense_voice_context_params {
    bool use_gpu;
    bool use_itn;
    bool flash_attn;
    int gpu_device;  // CUDA device
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

struct sense_voice_context {
    int64_t t_load_ms = 0;
    int64_t t_start_ms = 0;
    int32_t language_id = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    ggml_type itype =
            ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)

    silero_vad_model vad_model;
    sense_voice_model model;
    sense_voice_vocab vocab;

    sense_voice_context_params params;

    struct sense_voice_state *state = nullptr;
    ggml_backend_t backend = nullptr;
    std::string path_model;
};

struct sense_voice_full_params sense_voice_full_default_params(enum sense_voice_decoding_strategy strategy);
bool ggml_graph_compute_helper(ggml_backend_sched_t sched, struct ggml_cgraph * graph, int n_threads);


#endif//SENSEVOICE_CPP_COMMON_H
