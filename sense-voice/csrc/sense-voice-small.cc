//
// Created by lovemefan on 2024/7/19.
//
#include "common.h"
#include "sense-voice-small.h"


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