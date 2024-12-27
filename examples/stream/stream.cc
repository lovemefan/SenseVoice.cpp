#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>
#include "sense-voice.h"
#include "common-sdl.h"
#include "common.h"

static std::string to_timestamp(int64_t t, bool comma = false) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

struct sense_voice_stream_params {
    int32_t n_threads         = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors      = 1;
    int32_t capture_id        = -1;
    int32_t beam_size         = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t best_of           = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY).greedy.best_of;
    int32_t chunk_size        = 100;                       // ms
    int32_t max_nomute_chunks = 8000 / chunk_size;  // chunks
    int32_t min_mute_chunks   = 1000 / chunk_size;    // chunks

    bool no_context    = true;
    bool no_timestamps = false;
    bool use_gpu       = true;
    bool flash_attn    = false;
    bool debug_mode    = false;
    bool use_vad       = false;

    std::string language  = "auto";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};


void sense_voice_stream_usage(int /*argc*/, char ** argv, const sense_voice_stream_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N         [%-7d] [SenseVoice] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "            --chunk_size        [%-7d] vad chunk size(ms)\n",                                       params.chunk_size);
    fprintf(stderr, "  -mmc      --min-mute-chunks   [%-7d] When consecutive chunks are identified as silence\n",        params.min_mute_chunks);
    fprintf(stderr, "  -mnc      --max-nomute-chunks [%-7d] when the first non-silent chunk is too far away\n",          params.max_nomute_chunks);
    fprintf(stderr, "            --use_vad           [%-7s] when the first non-silent chunk is too far away\n",          params.use_vad ? "true" : "false");
    fprintf(stderr, "  -c ID,    --capture ID        [%-7d] [Device] capture device ID\n",                               params.capture_id);
    fprintf(stderr, "  -kc,      --keep-context      [%-7s] [IO] keep context between audio chunks\n",                   params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG     [%-7s] [SenseVoice] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME       [%-7s] [SenseVoice] model path\n",                                  params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME        [%-7s] [IO] text output file name\n",                               params.fname_out.c_str());
    fprintf(stderr, "  -ng,      --no-gpu            [%-7s] [SenseVoice] disable GPU inference\n",                       params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn        [%-7s] [SenseVoice]flash attention during inference\n",             params.flash_attn ? "true" : "false");
    fprintf(stderr, "\n");
}


static bool get_stream_params(int argc, char ** argv, sense_voice_stream_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            sense_voice_stream_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-debug"|| arg == "--debug-mode")    { params.debug_mode    = true; }
        else if (arg == "-mmc"  || arg == "--min-mute-chunks")   { params.min_mute_chunks   = std::stoi(argv[++i]); }
        else if (arg == "-mnc"  || arg == "--max-nomute-chunks") { params.max_nomute_chunks = std::stoi(argv[++i]); }
        else if (                  arg == "--use-vad")           { params.use_vad           = true; }
        else if (                  arg == "--chunk-size")        { params.chunk_size        = std::stoi(argv[++i]); }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sense_voice_stream_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}


int main(int argc, char** argv)
{
    sense_voice_stream_params params;
    if (get_stream_params(argc, argv, params) == false) return 1;
    const int n_sample_step = params.chunk_size * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const int keep_nomute_step = params.chunk_size * params.min_mute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const int max_nomute_step = params.chunk_size * params.max_nomute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;

    params.no_timestamps  = !params.use_vad;
    params.no_context |= params.use_vad;
    // params.max_tokens = 0;

    audio_async audio(params.chunk_size << 2);
    if (!audio.init(params.capture_id, SENSE_VOICE_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }
    audio.resume();

    struct sense_voice_context_params cparams = sense_voice_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    bool is_running = true;
    struct sense_voice_context *ctx = sense_voice_small_init_from_file_with_params(params.model.c_str(), cparams);
    std::vector<float> pcmf32_audio;
    std::vector<double> pcmf32;
    std::vector<double> pcmf32_tmp;  // 传递给模型用
    std::vector<int> prompt_tokens;
    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing samples (chunk = %d ms / max_nomute_chunk = %d / min_mute_chunk = %d), %d threads, timestamps = %d ...\n",
                __func__,
                params.chunk_size,
                params.max_nomute_chunks,
                params.min_mute_chunks,
                params.n_threads,
                params.no_timestamps ? 0 : 1);

        if (!params.use_vad) {
            fprintf(stderr, "%s: not use VAD, will print identified result per %d ms\n", __func__, params.chunk_size * params.max_nomute_chunks);
        } else {
            fprintf(stderr, "%s: using VAD, will print when mute time longer than %d ms or nomute time longer than %d ms\n", __func__, params.chunk_size * params.min_mute_chunks, params.chunk_size * params.max_nomute_chunks);
        }

        fprintf(stderr, "\n");
    }


    sense_voice_full_params wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
    {
        wparams.print_progress   = false;
        wparams.no_timestamps    = !params.no_timestamps;
        wparams.language         = params.language.c_str();
        wparams.n_threads        = params.n_threads;
        wparams.debug_mode       = params.debug_mode;

        wparams.greedy.best_of        = params.best_of;
        wparams.beam_search.beam_size = params.beam_size;
        wparams.in_stream = true;
    }

    int idenitified_floats = 0, R_new_chunk = 0, L_new_chunk = 0;
    // 只有vad会使用，但是会贯穿全程，只能定义在外面
    int L_nomute = -1, L_mute = -1, R_mute = -1;
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();
        if (!is_running) break;
        // process new audio
        // 获取新的音频，不论是否检测音频数据先把数据捞出来
        std::this_thread::sleep_for(std::chrono::milliseconds(params.chunk_size));
        audio.get(params.chunk_size, pcmf32_audio);
        // 转移到pcmf32中，直接识别pcmf32
        pcmf32.insert(pcmf32.end(), pcmf32_audio.begin(), pcmf32_audio.end());
        pcmf32_audio.clear();
        // 新的识别区间：[L_new_chunk, R_new_chunk)，是固定n_sample_step的整数倍
        // fprintf(stderr, "%s || new_audio_size: %d, cache_size: %d, L_new_chunk: %d, R_new_chunk: %d\n", __func__, pcmf32_audio.size(), pcmf32.size(), L_new_chunk, R_new_chunk);
        if(R_new_chunk + n_sample_step <= pcmf32.size())
        {
            L_new_chunk = R_new_chunk;
            R_new_chunk = int(pcmf32.size() / n_sample_step) * n_sample_step;
        }
        else continue;

        if (!params.use_vad) {
            // 识别当前已经识别到的文本恩
            if (sense_voice_full_parallel(ctx, wparams, pcmf32, R_new_chunk, params.n_processors) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 10;
            }
            printf("\33[2K\r");
            printf("%s", std::string(100, ' ').c_str());
            printf("\33[2K\r");
            printf("[%.2f-%.2f]", idenitified_floats / (SENSE_VOICE_SAMPLE_RATE * 1.0), (R_new_chunk + idenitified_floats) / (SENSE_VOICE_SAMPLE_RATE * 1.0));

            int last_id = 0;
            for(int i = 4; i < ctx->state->ids.size(); i++)
            {
                int id = ctx->state->ids[i];
                if (id != 0 && id != last_id) {
                    // printf("%d %d %s\n", i, id, ctx->vocab.id_to_token[id].c_str());
                    printf("%s", ctx->vocab.id_to_token[id].c_str());
                    last_id = id;
                }
            }

            // 时间长度太长了直接换行重新开始
            if (R_new_chunk >= max_nomute_step)
            {
                printf("\n");
                pcmf32_tmp = std::vector<double>(pcmf32.begin() + R_new_chunk, pcmf32.end());
                pcmf32 = pcmf32_tmp;
                idenitified_floats += R_new_chunk;
                L_new_chunk = R_new_chunk = 0;
            }
        }
        else
        {
            // 基于pcmf32.begin()是0来计算的，转化成全局坐标需要加上idenitified_floats
            for(int i = L_new_chunk; i < R_new_chunk; i += n_sample_step)
            {
                int R_this_chunk = i + n_sample_step;
                bool isnomute = vad_energy_zcr<double>(pcmf32.begin() + i, n_sample_step, SENSE_VOICE_SAMPLE_RATE);
                fprintf(stderr, "Mute || isnomute = %d, L_mute = %d, R_Mute = %d, L_nomute = %d, R_this_chunk = %d, keep_nomute_step = %d\n", isnomute, L_mute, R_mute, L_nomute, R_this_chunk, keep_nomute_step);
                if (L_nomute >= 0 && R_this_chunk - L_nomute >= max_nomute_step)
                {
                    int R_nomute = L_mute >= 0 ? L_mute : R_this_chunk;
                    // printf("3333: %d %d %d %d %d\n", L_nomute, R_nomute, L_mute, i, R_this_chunk);
                    pcmf32_tmp.resize(R_nomute - L_nomute);
                    std::copy(pcmf32.begin() + L_nomute, pcmf32.begin() + R_nomute, pcmf32_tmp.begin());
                    printf("[%.2f-%.2f]", (L_nomute + idenitified_floats) / (SENSE_VOICE_SAMPLE_RATE * 1.0), (R_nomute + idenitified_floats) / (SENSE_VOICE_SAMPLE_RATE * 1.0));
                    if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), params.n_processors) != 0) {
                        fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                        return 10;
                    }
                    int last_id = 0;
                    for(int i = 4; i < ctx->state->ids.size(); i++)
                    {
                        int id = ctx->state->ids[i];
                        if (id != 0 && id != last_id) {
                            // printf("%d %d %s\n", i, id, ctx->vocab.id_to_token[id].c_str());
                            printf("%s", ctx->vocab.id_to_token[id].c_str());
                            last_id = id;
                        }
                    }
                    printf("\n");
                    if (!isnomute) L_nomute = -1;
                    else if (R_mute >= 0) L_nomute = R_mute;
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
                        printf("[%.2f-%.2f]", (L_nomute + idenitified_floats) / (SENSE_VOICE_SAMPLE_RATE * 1.0), (L_mute + idenitified_floats) / (SENSE_VOICE_SAMPLE_RATE * 1.0));
                        if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), params.n_processors) != 0) {
                            fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                            return 10;
                        }
                        int last_id = 0;
                        for(int i = 4; i < ctx->state->ids.size(); i++)
                        {
                            int id = ctx->state->ids[i];
                            if (id != 0 && id != last_id) {
                                // printf("%d %d %s\n", i, id, ctx->vocab.id_to_token[id].c_str());
                                printf("%s", ctx->vocab.id_to_token[id].c_str());
                                last_id = id;
                            }
                        }
                        printf("\n");
                        if (!isnomute) L_nomute = -1;
                        else if (R_mute >= 0) L_nomute = R_mute;
                        else L_nomute = i;
                        L_mute = R_mute = -1;
                    }
                }
            }
            if (L_nomute < 0)
            {
                idenitified_floats += R_new_chunk;
                pcmf32_tmp = std::vector<double>(pcmf32.begin() + R_new_chunk, pcmf32.end());
                pcmf32 = pcmf32_tmp;
                L_new_chunk = R_new_chunk = 0;
                L_mute = R_mute = -1;
            }
            else
            {
                idenitified_floats += L_nomute;
                pcmf32_tmp = std::vector<double>(pcmf32.begin() + L_nomute, pcmf32.end());
                pcmf32 = pcmf32_tmp;
                L_new_chunk -= L_nomute;
                R_new_chunk -= L_nomute;
                L_mute = std::max(-1, L_mute - L_nomute);
                R_mute = std::max(-1, R_mute - L_nomute);
                L_nomute = 0;
            }
        }
        fflush(stdout);
    }
    audio.pause();
    // sense_voice_free(ctx);
    return 0;
}