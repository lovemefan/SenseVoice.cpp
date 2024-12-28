//
// Created by lovemefan on 2024/7/19.
//

#ifndef SENSEVOICE_CPP_SENSE_VOICE_H
#define SENSEVOICE_CPP_SENSE_VOICE_H

#include "common.h"
#include "sense-voice-encoder.h"




int sense_voice_lang_id(const char * lang);
const char * sense_voice_lang_str(int id);
struct sense_voice_context_params sense_voice_context_default_params();
struct sense_voice_context * sense_voice_small_init_from_file_with_params(const char * path_model, struct sense_voice_context_params params);
struct sense_voice_context * sense_voice_small_init_from_file_with_params_no_state(const char * path_model, struct sense_voice_context_params params);
struct sense_voice_context *sense_voice_init_with_params_no_state(const char *path_model, sense_voice_context_params params);
int sense_voice_full_parallel(struct sense_voice_context * ctx,
                              sense_voice_full_params &params,
                              std::vector<double> &samples,
                              int n_samples,
                              int n_processors);
void sense_voice_print_output(struct sense_voice_context * ctx, bool need_prefix, bool use_itn);
void sense_voice_free_state(struct sense_voice_state * state);
#endif//SENSEVOICE_CPP_SENSE_VOICE_H
