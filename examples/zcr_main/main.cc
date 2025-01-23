//
// Created by lovemefan on 2024/7/21.
//
#include "common.h"
#include "sense-voice.h"
#include <cmath>
#include <cstdint>
#include <thread>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


// command-line parameters
struct sense_voice_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t n_mel       = 80;
    int32_t best_of       = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx     = 0;
    int32_t chunk_size = 100;                         // ms
    int32_t max_nomute_chunks = 30000 / chunk_size;   // chunks
    int32_t min_mute_chunks = 1000 / chunk_size;      // chunks
    int32_t max_chunks_in_batch = 60000 / chunk_size; // chunks

    bool debug_mode      = false;
    bool no_prints       = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool use_itn         = false;

    std::string language  = "auto";
    std::string prompt;
    std::string model     = "models/ggml-base.en.bin";


    std::string openvino_encode_device = "CPU";


    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

};

static int sense_voice_has_coreml(void) {
#ifdef SENSE_VOICE_USE_COREML
    return 1;
#else
    return 0;
#endif
}

static int sense_voice_has_openvino(void) {
#ifdef SENSE_VOICE_USE_OPENVINO
    return 1;
#else
    return 0;
#endif
}

const char * sense_voice_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "SSSE3 = "     + std::to_string(ggml_cpu_has_ssse3())     + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";
    s += "COREML = "    + std::to_string(sense_voice_has_coreml()) + " | ";
    s += "OPENVINO = "  + std::to_string(sense_voice_has_openvino());

    return s.c_str();
}

static void sense_voice_print_usage(int /*argc*/, char ** argv, const sense_voice_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0.wav file1.wav ...\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n",                    params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n",                           params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n",   params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n",              params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N       [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n",           params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -of FNAME, --output-file FNAME [%-7s] output file path (without file extension)\n",      "");
    fprintf(stderr, "  -np,       --no-prints         [%-7s] do not print anything other than the results\n",   params.no_prints ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n",                                 params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n",                        params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect), support [`zh`, `en`, `yue`, `ja`, `ko`\n", params.language.c_str());
    fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt (max n_text_ctx/2 tokens)\n",       params.prompt.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] input WAV file path\n",                            "");
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n",  params.openvino_encode_device.c_str());
    fprintf(stderr, "  -ng,       --no-gpu            [%-7s] disable GPU\n",                                    params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,       --flash-attn        [%-7s] flash attention\n",                                params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -itn,      --use-itn           [%-7s] use itn\n",                                        params.use_itn ? "true" : "false");
    fprintf(stderr, "             --chunk_size        [%-7d] vad chunk size(ms)\n",                                params.chunk_size);
    fprintf(stderr, "  -mmc       --min-mute-chunks   [%-7d] When consecutive chunks are identified as silence\n", params.min_mute_chunks);
    fprintf(stderr, "  -mnc       --max-nomute-chunks [%-7d] when the first non-silent chunk is too far away\n",   params.max_nomute_chunks);
    fprintf(stderr, "\n");
}

struct sense_voice_print_user_data {
    const sense_voice_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

static char * sense_voice_param_turn_lowercase(char * in){
    int string_len = strlen(in);
    for (int i = 0; i < string_len; i++){
        *(in+i) = tolower((unsigned char)*(in+i));
    }
    return in;
}

static bool sense_voice_params_parse(int argc, char ** argv, sense_voice_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-"){
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            sense_voice_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")         { params.n_threads       = std::stoi(argv[++i]); }
        else if (arg == "-p"    || arg == "--processors")      { params.n_processors    = std::stoi(argv[++i]); }
        else if (arg == "-ot"   || arg == "--offset-t")        { params.offset_t_ms     = std::stoi(argv[++i]); }
        else if (arg == "-on"   || arg == "--offset-n")        { params.offset_n        = std::stoi(argv[++i]); }
        else if (arg == "-d"    || arg == "--duration")        { params.duration_ms     = std::stoi(argv[++i]); }
        else if (arg == "-mc"   || arg == "--max-context")     { params.max_context     = std::stoi(argv[++i]); }
        else if (arg == "-bo"   || arg == "--best-of")         { params.best_of         = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")       { params.beam_size       = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")       { params.audio_ctx       = std::stoi(argv[++i]); }
        else if (arg == "-debug"|| arg == "--debug-mode")      { params.debug_mode      = true; }
        else if (arg == "-of"   || arg == "--output-file")     { params.fname_out.emplace_back(argv[++i]); }
        else if (arg == "-np"   || arg == "--no-prints")       { params.no_prints       = false; }
        else if (arg == "-pp"   || arg == "--print-progress")  { params.print_progress  = true; }
        else if (arg == "-nt"   || arg == "--no-timestamps")   { params.no_timestamps   = true; }
        else if (arg == "-l"    || arg == "--language")        { params.language        = sense_voice_param_turn_lowercase(argv[++i]); }
        else if (                  arg == "--prompt")          { params.prompt          = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")           { params.model           = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")            { params.fname_inp.emplace_back(argv[++i]); }
        else if (arg == "-oved" || arg == "--ov-e-device")     { params.openvino_encode_device = argv[++i]; }
        else if (arg == "-ng"   || arg == "--no-gpu")          { params.use_gpu         = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")      { params.flash_attn      = true; }
        else if (arg == "-itn"  || arg == "--use-itn")         { params.use_itn         = true; }
        else if (arg == "-mmc"  || arg == "--min-mute-chunks")   { params.min_mute_chunks       = std::stoi(argv[++i]); }
        else if (arg == "-mnc"  || arg == "--max-nomute-chunks") { params.max_nomute_chunks     = std::stoi(argv[++i]); }
        else if (                  arg == "--chunk_size")        { params.chunk_size            = std::stoi(argv[++i]); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sense_voice_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

static bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v = 0;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        printf("fatal error");
                    }
                    printf("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("                                      ],\n");
        }
        printf("                                     ]\n");
        printf("                                     sum = %f\n", sum);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
           t->name, ggml_type_name(t->type), ggml_op_desc(t),
           src0->name, ggml_ne_string(src0).c_str(),
           src1 ? src1_str : "",
           ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
    }

    return true;
}

void sense_voice_free(struct sense_voice_context * ctx) {
    if (ctx) {
        ggml_free(ctx->model.ctx);

        ggml_backend_buffer_free(ctx->model.buffer);

        sense_voice_free_state(ctx->state);

        delete ctx->model.model->encoder;
        delete ctx->model.model;
        delete ctx;
    }
}

int main(int argc, char ** argv) {
    sense_voice_params params;

    if (!sense_voice_params_parse(argc, argv, params)) {
        sense_voice_print_usage(argc, argv, params);
        return 1;
    }

    // remove non-existent files
    for (auto it = params.fname_inp.begin(); it != params.fname_inp.end();) {
        const auto fname_inp = it->c_str();

        if (*it != "-" && !is_file_exist(fname_inp)) {
            fprintf(stderr, "error: input file not found '%s'\n", fname_inp);
            it = params.fname_inp.erase(it);
            continue;
        }

        it++;
    }

    if (params.fname_inp.empty()) {
        fprintf(stderr, "error: no input files specified\n");
        sense_voice_print_usage(argc, argv, params);
        return 2;
    }

    if (params.language != "auto" && sense_voice_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        sense_voice_print_usage(argc, argv, params);
        exit(0);
    }

    // sense-voice init

    struct sense_voice_context_params cparams = sense_voice_context_default_params();

    callback_data cb_data;

    cparams.cb_eval = ggml_debug;
    cparams.cb_eval_user_data = &cb_data;

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    cparams.use_itn    = params.use_itn;

    struct sense_voice_context * ctx = sense_voice_small_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize sense voice context\n");
        return 3;
    }

    ctx->language_id = sense_voice_lang_id(params.language.c_str());

    for (int f = 0; f < (int) params.fname_inp.size(); ++f) {
        const auto fname_inp = params.fname_inp[f];
        const auto fname_out = f < (int) params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

        std::vector<double> pcmf32;               // mono-channel F32 PCM

        int sample_rate;
        if (!::load_wav_file(fname_inp.c_str(), &sample_rate, pcmf32)) {
            fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp.c_str());
            continue;
        }

        if (!params.no_prints) {
            // print system information
            fprintf(stderr, "\n");
            fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                    params.n_threads*params.n_processors, std::thread::hardware_concurrency(), sense_voice_print_system_info());

            // print some info about the processing
            fprintf(stderr, "\n");
            fprintf(stderr, "%s: processing audio (%d samples, %.5f sec) , %d threads, %d processors, lang = %s...\n",
                    __func__,  int(pcmf32.size()), float(pcmf32.size())/sample_rate,
                    params.n_threads, params.n_processors,
                    params.language.c_str());
            ctx->state->duration = float(pcmf32.size())/sample_rate;
            fprintf(stderr, "\n");
        }

        {
            sense_voice_full_params wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
            wparams.strategy = (params.beam_size > 1 ) ? SENSE_VOICE_SAMPLING_BEAM_SEARCH : SENSE_VOICE_SAMPLING_GREEDY;
            wparams.print_progress   = params.print_progress;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms        = params.offset_t_ms;
            wparams.duration_ms      = params.duration_ms;
            wparams.debug_mode       = params.debug_mode;
            wparams.greedy.best_of        = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;
            wparams.no_timestamps    = params.no_timestamps;

            std::vector<double> pcmf32_tmp;           // 记录需要解析的段落
            int32_t L_nomute = -1, L_mute = -1, R_mute = -1;  // [L_nomute, R_nomute)永远为需要解析的段落，[L_mute, R_mute)永远为最近一段静音空挡
            const int n_sample_step = params.chunk_size * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
            const int keep_nomute_step = params.chunk_size * params.min_mute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
            const int max_nomute_step = params.chunk_size * params.max_nomute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
            std::vector<double> pcmf32_chunk;         // 作为临时缓冲区
            pcmf32_chunk.resize(n_sample_step);
            // const bool use_vad = (n_samples_step <= 0);
            for(int i = 0; i < pcmf32.size(); i += n_sample_step)
            {
                int R_this_chunk = std::min(i + n_sample_step, int(pcmf32.size()));
                bool isnomute = vad_energy_zcr<double>(pcmf32.begin() + i, R_this_chunk - i, SENSE_VOICE_SAMPLE_RATE);

                // fprintf(stderr, "Mute || L_mute = %d, R_Mute = %d, L_nomute = %d, R_this_chunk = %d, keep_nomute_step = %d\n", L_mute, R_mute, L_nomute, R_this_chunk, keep_nomute_step);
                if (L_nomute >= 0 && R_this_chunk - L_nomute >= max_nomute_step)
                {
                    int R_nomute = L_mute >= 0 && L_mute >= L_nomute ? L_mute : R_this_chunk;
                    pcmf32_tmp.resize(R_nomute - L_nomute);
                    std::copy(pcmf32.begin() + L_nomute, pcmf32.begin() + R_nomute, pcmf32_tmp.begin());
                    printf("[%.2f-%.2f]", L_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0), R_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0));
                    if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), params.n_processors) != 0) {
                        fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                        return 10;
                    }
                    sense_voice_print_output(ctx, true, params.use_itn);
                    if (!isnomute) L_nomute = -1;
                    else if (R_mute >= 0 && L_mute >= L_nomute) L_nomute = R_mute;
                    else L_nomute = i;
                    L_mute = R_mute = -1;
                    continue;
                }
                if (isnomute)
                {
                    if (L_nomute < 0) L_nomute = i;
                }
                else
                {
                    if (R_mute != i) L_mute = i;
                    R_mute = R_this_chunk;
                    if (L_mute >= L_nomute && L_nomute >= 0 && R_this_chunk - L_mute >= keep_nomute_step)
                    {
                        // printf("2222: %d %d %d %d %d\n", L_nomute, R_nomute, L_mute, i, R_this_chunk);
                        pcmf32_tmp.resize(L_mute - L_nomute);
                        std::copy(pcmf32.begin() + L_nomute, pcmf32.begin() + L_mute, pcmf32_tmp.begin());
                        printf("[%.2f-%.2f]", L_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0), L_mute / (SENSE_VOICE_SAMPLE_RATE * 1.0));
                        if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), params.n_processors) != 0) {
                            fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                            return 10;
                        }
                        sense_voice_print_output(ctx, true, params.use_itn);
                        if (!isnomute) L_nomute = -1;
                        else if (R_mute >= 0) L_nomute = R_mute;
                        else L_nomute = i;
                        L_mute = R_mute = -1;
                    }
                }
            }
            // 最后一段
            if(L_nomute >= 0)
            {
                int R_nomute = pcmf32.size();
                printf("[%.2f-%.2f]", L_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0), R_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0));
                pcmf32_tmp.resize(R_nomute - L_nomute);
                std::copy(pcmf32.begin() + L_nomute, pcmf32.begin() + R_nomute, pcmf32_tmp.begin());
                if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), params.n_processors) != 0) {
                    fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                    return 10;
                } 
                sense_voice_print_output(ctx, true, params.use_itn);
                L_nomute = L_mute = R_mute = -1;
            }
        }
    }
    sense_voice_free(ctx);
    return 0;
}