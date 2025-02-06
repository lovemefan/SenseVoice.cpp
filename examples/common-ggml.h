//
// Created by lovemefan on 2024/8/27.
//

#pragma once
#include <gguf.h>
#include <ggml.h>
#include <fstream>
#include <vector>
#include <cstring>

enum ggml_ftype ggml_parse_ftype(const char * str);

void ggml_print_ftypes(FILE * fp = stderr);

bool sense_voice_ggml_quantize0(
        ggml_context * ctx,
        gguf_context * gguf_input,
        const std::string & fname_inp,
        const std::string & fname_out,
        const ggml_ftype ftype,
        const int nthread,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);