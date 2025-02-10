// Copyright  2024  lovemefan
// Created by lovemefan on 2023/10/3.
//

#include "sense-voice-frontend.h"
#include <algorithm>
#include <cassert>
#include "ThreadPool.h"
#include "log-mel-filter-bank.h"

#define M_2PI 6.283185307179586476925286766559005
#define SIN_COS_N_COUNT 512

// In FFT, we frequently use sine and cosine operations with the same values.
// We can use precalculated values to speed up the process.
std::vector<int> ip_(2 + std::sqrt(SIN_COS_N_COUNT / 2));
std::vector<double> w_(SIN_COS_N_COUNT / 2);

// see fftsg.cc
void rdft(int n, int isgn, double *a, int *ip, double *w);

static void rfft(const std::vector<double> &in) {
  int n = in.size();
  rdft(n, 1, (double *)in.data(), ip_.data(), (double *)w_.data());

}

inline int round_to_nearest_power_two(int n) {
  // copied from kaldi/src/base/kaldi-math.cc
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

static bool hamming_window(int length, bool periodic,
                           std::vector<double> &output) {
  if (output.size() < static_cast<size_t>(length)) {
    output.resize(length);
  }
  int offset = -1;
  if (periodic) {
    offset = 0;
  }
  for (int i = 0; i < length; i++) {
    output[i] = 0.54 - 0.46 * cosf((M_2PI * i) / (length + offset));
  }

  return true;
}


static void fbank_feature_worker_thread(int ith,
                                        const std::vector<double> &hamming,
                                        const std::vector<double> &samples,
                                        int n_samples, int frame_size,
                                        int frame_step, int n_threads,
                                        sense_voice_feature &mel) {
  // make sure n_fft == 1 + (sense_voice_N_FFT / 2), bin_0 to bin_nyquist
  int i = ith;

  std::vector<double> window;
  const int padded_window_size = round_to_nearest_power_two(frame_size);
  window.resize(padded_window_size);

  // calculate FFT only when fft_in are not all zero
  int n_fft = std::min(n_samples / frame_step + 1, mel.n_len);
  for (; i < n_fft; i += n_threads) {
    const int offset = i * frame_step;

    std::copy(samples.begin() + offset, samples.begin() + offset + frame_size,
              window.begin());

    {
        // init window default 0, initialization values may result in NaN on arm cpu.
        for (int k = frame_size; k < window.size(); k++) {
            window[k] = 0;
        }
    }
    // remove dc offset
    {
      double sum = 0;
      for (int32_t k = 0; k < frame_size; ++k) {
        sum += window[k];
      }
      double mean = sum / frame_size;
      for (int32_t k = 0; k < frame_size; ++k) {
        window[k] -= mean;
      }
    }
    // pre-emphasis
    {
      for (int32_t k = frame_size - 1; k > 0; --k) {
        window[k] -= PREEMPH_COEFF * window[k - 1];
      }
      window[0] -= PREEMPH_COEFF * window[0];
    }

    // apply Hamming window
    {
      for (int j = 0; j < frame_size; j++) {
        window[j] *= hamming[j];
      }
    }

    // FFT
    // window is input and output
    rfft(window);


    // Calculate modulus^2 of complex numbers，Power Spectrum
    // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes
    // inference quality problem? Interesting.
    for (int j = 0; j < padded_window_size / 2; j++) {
      window[j] = (window[2 * j + 0] * window[2 * j + 0] +
                   window[2 * j + 1] * window[2 * j + 1]);
    }

    // log-Mel filter bank energies aka: "fbank"
    {
      auto num_fft_bins = padded_window_size / 2;
      int n_mel = mel.n_mel;
      for (int j = 0; j < n_mel; j++) {
        double sum = 0.0;
        for (int k = 0; k < num_fft_bins; k++) {
          sum += window[k] * LogMelFilterMelArray[j * num_fft_bins + k];
        }

        sum = log(sum > 1.19e-7 ? sum : 1.19e-7);

        mel.data[i * n_mel + j] = static_cast<float>(sum);
      }
    }
  }
}

bool fbank_lfr_cmvn_feature(const std::vector<double> &samples,
                            const int n_samples, const int frame_size,
                            const int frame_step, const int n_feats,
                            const int n_threads, const bool debug,
                            sense_voice_cmvn &cmvn, sense_voice_feature &feats) {
  //    const int64_t t_start_us = ggml_time_us();

  const int32_t n_frames_per_ms = SENSE_VOICE_SAMPLE_RATE * 0.001f;
  feats.n_mel = n_feats;
  feats.n_len = 1 + ((n_samples - frame_size * n_frames_per_ms) /
                   (frame_step * n_frames_per_ms));
  feats.data.resize(feats.n_mel * feats.n_len);

  std::vector<double> hamming;
  hamming_window(frame_size * n_frames_per_ms, true, hamming);

  {
    if (n_threads > 1) {
        ThreadPool pool(n_threads);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            pool.enqueue(fbank_feature_worker_thread, iw + 1, std::cref(hamming),
                         samples, n_samples, frame_size * n_frames_per_ms,
                         frame_step * n_frames_per_ms, n_threads, std::ref(feats));
        }
    }

    // main thread
    fbank_feature_worker_thread(0, hamming, samples, n_samples,
                                frame_size * n_frames_per_ms,
                                frame_step * n_frames_per_ms, n_threads, feats);
  }

  if (debug) {
      auto &mel = feats.data;
      std::ofstream outFile("fbank_lfr_cmvn_feature.json");
      outFile << "[";
      for (uint64_t i = 0; i < mel.size() - 1; i++) {
          outFile << mel[i] << ", ";
      }
      outFile << mel[mel.size() - 1] << "]";
      outFile.close();
  }

  std::vector<std::vector<float>> out_feats;

  // tapply lrf, merge lfr_m frames as one,lfr_n frames per window
  // ref:
  // https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/paraformer.cpp#L409-L440
  int T = feats.n_len;
  int lfr_m = feats.lfr_m;  // 7
  int lfr_n = feats.lfr_n;  // 6
  int T_lrf = ceil(1.0 * T / feats.lfr_n);
  int left_pad = (feats.lfr_m - 1) / 2;
  int left_pad_offset = (lfr_m - left_pad) * feats.n_mel;
  // Merge lfr_m frames as one,lfr_n frames per window
  T = T + (lfr_m - 1) / 2;
  std::vector<float> p;
  for (int i = 0; i < T_lrf; i++) {
    // the first frames need left padding
    if (i == 0) {
      // left padding
      for (int j = 0; j < left_pad; j++) {
        p.insert(p.end(), feats.data.begin(), feats.data.begin() + feats.n_mel);
      }
      p.insert(p.end(), feats.data.begin(), feats.data.begin() + left_pad_offset);
      out_feats.push_back(p);
      p.clear();
    } else {
      if (lfr_m <= T - i * lfr_n) {
        p.insert(p.end(), feats.data.begin() + (i * lfr_n - left_pad) * feats.n_mel,
                   feats.data.begin() + (i * lfr_n - left_pad + lfr_m) * feats.n_mel);
        out_feats.push_back(p);
        p.clear();
      } else {
        // Fill to lfr_m frames at last window if less than lfr_m frames  (copy
        // last frame)
        int num_padding = lfr_m - (T - i * lfr_n);
        for (int j = 0; j < (feats.n_len - i * lfr_n); j++) {
          p.insert(p.end(),
                     feats.data.begin() + (i * lfr_n - left_pad) * feats.n_mel,
                     feats.data.end());
        }
        for (int j = 0; j < num_padding; j++) {
          p.insert(p.end(), feats.data.end() - feats.n_mel, feats.data.end());
        }
        out_feats.push_back(p);
        p.clear();
      }
    }
  }
  feats.data.resize(T_lrf * feats.lfr_m * feats.n_mel);
  // apply cvmn
  for (int i = 0; i < T_lrf; i++) {
    for (int j = 0; j < feats.lfr_m * feats.n_mel; j++) {
        feats.data[i * feats.lfr_m * feats.n_mel + j] = (out_feats[i][j] + cmvn.cmvn_means[j]) * cmvn.cmvn_vars[j];
    }
  }
  return true;
}

bool load_wav_file(const char *filename, int32_t *sampling_rate,
                   std::vector<double> &data) {
  struct WaveHeader header {};

  std::ifstream is(filename, std::ifstream::binary);
  is.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!is) {
    std::cout << "Failed to read " << filename;
    return false;
  }

  if (!header.Validate()) {
    return false;
  }

  header.SeekToDataChunk(is);
  if (!is) {
    return false;
  }

  *sampling_rate = header.sample_rate;
  // header.subchunk2_size contains the number of bytes in the data.
  // As we assume each sample contains two bytes, so it is divided by 2 here
  auto speech_len = header.subchunk2_size / 2;
  data.resize(speech_len);

  auto speech_buff = (int16_t *)malloc(sizeof(int16_t) * speech_len);

  if (speech_buff) {
    memset(speech_buff, 0, sizeof(int16_t) * speech_len);
    is.read(reinterpret_cast<char *>(speech_buff), header.subchunk2_size);
    if (!is) {
      std::cout << "Failed to read " << filename;
      return false;
    }

//    float scale = 32768;
    float scale = 1.0;
    for (int32_t i = 0; i != speech_len; ++i) {
      data[i] = (double)speech_buff[i] / scale;
    }
    free(speech_buff);
    return true;
  } else {
    free(speech_buff);
    return false;
  }

}

