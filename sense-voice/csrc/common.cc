//
// Created by lovemefan on 2024/7/21.
//
//
// logging
//

#include "common.h"
#include <stdarg.h>
#ifdef __GNUC__
#ifdef __MINGW32__
#define SENSEVOICE_ATTRIBUTE_FORMAT(...) \
  __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define SENSEVOICE_ATTRIBUTE_FORMAT(...) \
  __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define SENSEVOICE_ATTRIBUTE_FORMAT(...)
#endif

void sense_voice_log_callback_default(ggml_log_level level,
                                             const char *text, void *user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

GGML_ATTRIBUTE_FORMAT(2, 3)
void sense_voice_log_internal(ggml_log_level level, const char *format,
                                     ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    int len = vsnprintf(buffer, 1024, format, args);
    if (len < 1024) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char *buffer2 = new char[len + 1];
        vsnprintf(buffer2, len + 1, format, args);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args);
}


struct sense_voice_full_params sense_voice_full_default_params(enum sense_voice_decoding_strategy strategy) {
    struct sense_voice_full_params result = {
            /*.strategy          =*/ strategy,

            /*.n_threads         =*/ std::min(4, (int32_t) std::thread::hardware_concurrency()),
            /* language          =*/ "auto",
            /*.n_max_text_ctx    =*/ 16384,
            /*.offset_ms         =*/ 0,
            /*.duration_ms       =*/ 0,

            /*.no_context        =*/ true,
            /*.no_timestamps     =*/ false,
            /*.print_progress    =*/ true,
            /*.print_timestamps  =*/ true,


            /*.debug_mode        =*/ false,
            /* audio_ctx         =*/ 0,

            /*.greedy            =*/ {
                    /*.best_of   =*/ -1,
            },

            /*.beam_search      =*/ {
                    /*.beam_size =*/ -1,
            },

            /*.progress_callback           =*/ nullptr,
            /*.progress_callback_user_data =*/ nullptr,

    };

    switch (strategy) {
        case SENSE_VOICE_SAMPLING_GREEDY:
        {
            result.greedy = {
                    /*.best_of   =*/ 5,
            };
        } break;
        case SENSE_VOICE_SAMPLING_BEAM_SEARCH:
        {
            result.beam_search = {
                    /*.beam_size =*/ 5
            };
        } break;
    }

    return result;
}

bool ggml_graph_compute_helper(
        ggml_backend_sched_t   sched,
        struct ggml_cgraph * graph,
        int   n_threads) {

    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        auto * fn_set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn_set_n_threads) {
            fn_set_n_threads(backend, n_threads);
        }
    }


    bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_reset(sched);
    return t;
}