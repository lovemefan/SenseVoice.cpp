//
// Created by lovemefan on 2024/7/19.
//

#include "sense-voice-small.h"
#include "common.h"

#ifdef __GNUC__
#ifdef __MINGW32__
#define SENSE_VOICE_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define SENSE_VOICE_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define SENSE_VOICE_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

SENSE_VOICE_ATTRIBUTE_FORMAT(2, 3)
static void whisper_log_internal        (ggml_log_level level, const char * format, ...);
static void whisper_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define SENSE_VOICE_LOG_ERROR(...) whisper_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define SENSE_VOICE_LOG_WARN(...)  whisper_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define SENSE_VOICE_LOG_INFO(...)  whisper_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)

// define this to enable verbose trace logging - useful for debugging purposes
//#define SENSE_VOICE_DEBUG

#if defined(SENSE_VOICE_DEBUG)
#define SENSE_VOICE_LOG_DEBUG(...) whisper_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
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