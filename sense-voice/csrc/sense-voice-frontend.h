//
// Created by lovemefan on 2024/7/17.
//
#pragma once
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"

#define SENSE_VOICE_SAMPLE_RATE 16000
#define PREEMPH_COEFF 0.97

struct sense_voice_feature {
  int n_len;
  int n_mel=80;
  float mel_low_freq = 31.748642f;
  float mel_high_freq = 2840.03784f;
  float vtln_high = -500.0f;
  float vtln_low = 100.0f;
  int lfr_n = 6;
  int lfr_m = 7;
  int32_t frame_size = 25;
  int32_t frame_step = 10;
  std::vector<float> data;
  std::vector<float> input_data;
  ggml_context * ctx = nullptr;
  ggml_tensor * tensor = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
};

struct sense_voice_cmvn {
  std::vector<float> cmvn_means;
  std::vector<float> cmvn_vars;
};

struct WaveHeader {
  bool Validate() const {
    //                 F F I R
    if (chunk_id != 0x46464952) {
      printf("Expected chunk_id RIFF. Given: 0x%08x\n", chunk_id);
      return false;
    }
    //               E V A W
    if (format != 0x45564157) {
      printf("Expected format WAVE. Given: 0x%08x\n", format);
      return false;
    }

    if (subchunk1_id != 0x20746d66) {
      printf("Expected subchunk1_id 0x20746d66. Given: 0x%08x\n", subchunk1_id);
      return false;
    }

    if (subchunk1_size != 16) {  // 16 for PCM
      printf("Expected subchunk1_size 16. Given: %d\n", subchunk1_size);
      return false;
    }

    if (audio_format != 1) {  // 1 for PCM
      printf("Expected audio_format 1. Given: %d\n", audio_format);
      return false;
    }

    if (num_channels != 1) {  // we support only single channel for now
      printf("Expected single channel. Given: %d\n", num_channels);
      return false;
    }
    if (byte_rate != (sample_rate * num_channels * bits_per_sample / 8)) {
      return false;
    }

    if (block_align != (num_channels * bits_per_sample / 8)) {
      return false;
    }

    if (bits_per_sample != 16) {  // we support only 16 bits per sample
      printf("Expected bits_per_sample 16. Given: %d\n", bits_per_sample);
      return false;
    }
    return true;
  }

  // See https://en.wikipedia.org/wiki/WAV#Metadata and
  // https://www.robotplanet.dk/audio/wav_meta_data/riff_mci.pdf
  void SeekToDataChunk(std::istream &is) {
    //                              a t a d
    while (is && subchunk2_id != 0x61746164) {
      // const char *p = reinterpret_cast<const char *>(&subchunk2_id);
      // printf("Skip chunk (%x): %c%c%c%c of size: %d\n", subchunk2_id, p[0],
      //        p[1], p[2], p[3], subchunk2_size);
      is.seekg(subchunk2_size, std::istream::cur);
      is.read(reinterpret_cast<char *>(&subchunk2_id), sizeof(int32_t));
      is.read(reinterpret_cast<char *>(&subchunk2_size), sizeof(int32_t));
    }
  }

  int32_t chunk_id;
  int32_t chunk_size;
  int32_t format;
  int32_t subchunk1_id;
  int32_t subchunk1_size;
  int16_t audio_format;
  int16_t num_channels;
  int32_t sample_rate;
  int32_t byte_rate;
  int16_t block_align;
  int16_t bits_per_sample;
  int32_t subchunk2_id;    // a tag of this chunk
  int32_t subchunk2_size;  // size of subchunk2
};

bool load_wav_file(const char *filename, int32_t *sampling_rate,
                   std::vector<float> &data);
bool fbank_lfr_cmvn_feature(const std::vector<double> &samples,
                            const int n_samples, const int frame_size,
                            const int frame_step, const int n_feats,
                            const int n_threads, const bool debug,
                            sense_voice_cmvn &cmvn, sense_voice_feature &feats);
bool load_wav_file(const char *filename, int32_t *sampling_rate,
                   std::vector<double> &data);


template<typename T>
bool vad_energy_zcr(const typename std::vector<T>::const_iterator& pcmf32, size_t siz, int sample_rate, T energy_threshold = 0.01, T zcr_threshold = 0.2, bool verbose = false)
{
    const int frame_size = 256; // 16ms at 16kHz
    const int frame_shift = 128; // 50% overlap
    
    if (siz < frame_size) return false;

    int num_frames = (siz - frame_size) / frame_shift + 1;
    std::vector<T> energies(num_frames);
    std::vector<T> zcrs(num_frames);

    // Calculate short-time energy and zero-crossing rate for each frame
    for (int f = 0; f < num_frames; f++) {
        T energy = 0.0f;
        int zcr = 0;
        
        int frame_start = f * frame_shift;
        
        // Calculate energy
        for (int i = 0; i < frame_size; i++) {
            T sample = pcmf32[frame_start + i];
            energy += sample * sample;
        }
        energy /= frame_size;
        energies[f] = energy;
        
        // Calculate zero-crossing rate
        for (int i = 1; i < frame_size; i++) {
            if ((pcmf32[frame_start + i - 1] * pcmf32[frame_start + i]) < 0)
                zcr++;
        }
        zcrs[f] = (T)zcr / frame_size;
    }

    // Get average energy and ZCR
    T avg_energy = 0.0f;
    T avg_zcr = 0.0f;
    for (int f = 0; f < num_frames; f++) {
        avg_energy += energies[f];
        avg_zcr += zcrs[f];
    }
    avg_energy /= num_frames;
    avg_zcr /= num_frames;

    if (verbose) {
        fprintf(stderr, "%s: avg_energy: %f, avg_zcr: %f, energy_threshold: %f, zcr_threshold: %f\n", 
                __func__, avg_energy, avg_zcr, energy_threshold, zcr_threshold);
    }

    // Voice activity is detected if either energy or ZCR exceeds their thresholds
    return (avg_energy > energy_threshold) || (avg_zcr > zcr_threshold);
}
