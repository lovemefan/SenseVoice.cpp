// Copyright  2024  lovemefan
// Created by lovemefan on 2023/10/3.
//

#include "sense-voice-frontend.h"
#include <algorithm>
#include <cassert>
#include <string>
#include "ThreadPool.h"
#include "log-mel-filter-bank.h"

#define M_2PI 6.283185307179586476925286766559005
#define SIN_COS_N_COUNT 512

// In FFT, we frequently use sine and cosine operations with the same values.
// We can use precalculated values to speed up the process.
std::vector<int32_t> ip_(2 + std::sqrt(SIN_COS_N_COUNT / 2));
std::vector<float> w_(SIN_COS_N_COUNT / 2);

// see fftsg.cc
void rdft(int n, int isgn, double *a, int *ip, double *w);

static void rfft(const std::vector<float> &in) {
  int32_t n = in.size();
  rdft(n, 1, (double *)in.data(), ip_.data(), (double *)w_.data());
}

inline int32_t round_to_nearest_power_two(int32_t n) {
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
                           std::vector<float> &output) {
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

void load_cmvn(const char *filename, sense_voice_cmvn &cmvn) {
  std::ifstream cmvn_stream(filename);
  if (!cmvn_stream.is_open()) {
    std::cout << "Failed to open file: " << filename;
    exit(-1);
  }
  std::string line;

  while (getline(cmvn_stream, line)) {
    std::istringstream iss(line);
    std::vector<std::string> line_item{std::istream_iterator<std::string>{iss},
                                       std::istream_iterator<std::string>{}};
    if (line_item[0] == "<AddShift>") {
      getline(cmvn_stream, line);
      std::istringstream means_lines_stream(line);
      std::vector<std::string> means_lines{
          std::istream_iterator<std::string>{means_lines_stream},
          std::istream_iterator<std::string>{}};
      if (means_lines[0] == "<LearnRateCoef>") {
        for (int j = 3; j < means_lines.size() - 1; j++) {
          cmvn.cmvn_means.push_back(stof(means_lines[j]));
        }
        continue;
      }
    } else if (line_item[0] == "<Rescale>") {
      getline(cmvn_stream, line);
      std::istringstream vars_lines_stream(line);
      std::vector<std::string> vars_lines{
          std::istream_iterator<std::string>{vars_lines_stream},
          std::istream_iterator<std::string>{}};
      if (vars_lines[0] == "<LearnRateCoef>") {
        for (int j = 3; j < vars_lines.size() - 1; j++) {
          cmvn.cmvn_vars.push_back(stof(vars_lines[j]));
        }
        continue;
      }
    }
  }
}

static void lrf_cmvn_worker_thread(
    int ith, sense_voice_feature &mel, sense_voice_cmvn &cmvn,
    std::vector<std::vector<float>> &out_feats) {
  // tapply lrf, merge lfr_m frames as one,lfr_n frames per window
  // ref:
  // https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/sense_voice.cpp#L409-L440
  int T = mel.n_len;
  int lfr_m = mel.lfr_m;  // 7
  int lfr_n = mel.lfr_n;  // 6
  int T_lrf = ceil(1.0 * T / mel.lfr_n);
  int left_pad = (mel.lfr_m - 1) / 2;
  int left_pad_offset = (lfr_m - left_pad) * mel.n_mel;
  // Merge lfr_m frames as one,lfr_n frames per window
  T = T + (lfr_m - 1) / 2;
  std::vector<float> p;
  for (int i = 0; i < T_lrf; i++) {
    // the first frames need left padding
    if (i == 0) {
      // left padding
      for (int j = 0; j < left_pad; j++) {
        p.insert(p.end(), mel.data.begin(), mel.data.begin() + mel.n_mel);
      }
      p.insert(p.end(), mel.data.begin(), mel.data.begin() + left_pad_offset);
      out_feats.emplace_back(p);
      p.clear();
    } else {
      if (lfr_m <= T - i * lfr_n) {
        p.insert(p.end(), mel.data.begin() + (i * lfr_n - left_pad) * mel.n_mel,
                 mel.data.begin() + (i * lfr_n - left_pad + lfr_m) * mel.n_mel);
        out_feats.emplace_back(p);
        p.clear();
      } else {
        // Fill to lfr_m frames at last window if less than lfr_m frames  (copy
        // last frame)
        int num_padding = lfr_m - (T - i * lfr_n);
        for (int j = 0; j < (mel.n_len - i * lfr_n); j++) {
          p.insert(p.end(),
                   mel.data.begin() + (i * lfr_n - left_pad) * mel.n_mel,
                   mel.data.end());
        }
        for (int j = 0; j < num_padding; j++) {
          p.insert(p.end(), mel.data.end() - mel.n_mel, mel.data.end());
        }
        out_feats.emplace_back(p);
        p.clear();
      }
    }
  }

  // apply cvmn
  for (auto &out_feat : out_feats) {
    for (int j = 0; j < cmvn.cmvn_means.size(); j++) {
      out_feat[j] = (out_feat[j] + cmvn.cmvn_means[j]) * cmvn.cmvn_vars[j];
    }
  }
}

static void fbank_feature_worker_thread(int ith,
                                        const std::vector<float> &hamming,
                                        const std::vector<float> &samples,
                                        int n_samples, int frame_size,
                                        int frame_step, int n_threads,
                                        sense_voice_feature &mel) {
  // make sure n_fft == 1 + (sense_voice_N_FFT / 2), bin_0 to bin_nyquist
  int i = ith;

  std::vector<float> window;
  const int padded_window_size = round_to_nearest_power_two(frame_size);
  window.resize(padded_window_size);
  // calculate FFT only when fft_in are not all zero
  int n_fft = std::min(n_samples / frame_step + 1, mel.n_len);
  for (; i < n_fft; i += n_threads) {
    const int offset = i * frame_step;

    std::copy(samples.begin() + offset, samples.begin() + offset + frame_size,
              window.begin());

    // remove dc offset
    {
      float sum = 0;
      for (int32_t i = 0; i != frame_size; ++i) {
        sum += window[i];
      }
      float mean = sum / frame_size;
      for (int32_t i = 0; i != frame_size; ++i) {
        window[i] -= mean;
      }
    }
    // pre-emphasis
    {
      for (int32_t i = frame_size - 1; i > 0; --i) {
        window[i] -= PREEMPH_COEFF * window[i - 1];
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
    for (int j = 0; j < padded_window_size; j++) {
      window[j] = (window[2 * j + 0] * window[2 * j + 0] +
                   window[2 * j + 1] * window[2 * j + 1]);
    }

    // log-Mel filter bank energies aka: "fbank"
    {
      auto num_fft_bins = padded_window_size / 2;
      int n_mel = mel.n_mel;
      for (int j = 0; j < n_mel; j++) {
        float sum = 0.0;
        for (int k = 0; k < num_fft_bins; k++) {
          sum += window[k] * LogMelFilterMelArray[j * num_fft_bins + k];
        }

        sum = log(sum > 1e-10 ? sum : 1e-10);

        mel.data[i * n_mel + j] = sum;
      }
    }
  }
}

bool fbank_lfr_cmvn_feature(const std::vector<float> &samples,
                            const int n_samples, const int frame_size,
                            const int frame_step, const int n_mel,
                            const int n_threads, const bool debug,
                            sense_voice_cmvn &cmvn, sense_voice_feature &mel) {
  //    const int64_t t_start_us = ggml_time_us();

  const int32_t n_frames_per_ms = SENSE_VOICE_SAMPLE_RATE * 0.001f;
  mel.n_mel = n_mel;
  mel.n_len = 1 + ((n_samples - frame_size * n_frames_per_ms) /
                   (frame_step * n_frames_per_ms));
  mel.data.resize(mel.n_mel * mel.n_len);

  std::vector<float> hamming;
  hamming_window(frame_size * n_frames_per_ms, true, hamming);

  {
    ThreadPool pool(n_threads);
    for (int iw = 0; iw < n_threads - 1; ++iw) {
      pool.enqueue(fbank_feature_worker_thread, iw + 1, std::cref(hamming),
                   samples, n_samples, frame_size * n_frames_per_ms,
                   frame_step * n_frames_per_ms, n_threads, std::ref(mel));
    }

    // main thread
    fbank_feature_worker_thread(0, hamming, samples, n_samples,
                                frame_size * n_frames_per_ms,
                                frame_step * n_frames_per_ms, n_threads, mel);
  }

  //    if (debug) {
  //        std::ofstream outFile("fbank_lfr_cmvn_feature.json");
  //        outFile << "[";
  //        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
  //            outFile << mel.data[i] << ", ";
  //        }
  //        outFile << mel.data[mel.data.size() - 1] << "]";
  //        outFile.close();
  //    }

  std::vector<std::vector<float>> out_feats;

  // tapply lrf, merge lfr_m frames as one,lfr_n frames per window
  // ref:
  // https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/sense_voice.cpp#L409-L440
  int T = mel.n_len;
  int lfr_m = mel.lfr_m;  // 7
  int lfr_n = mel.lfr_n;  // 6
  int T_lrf = ceil(1.0 * T / mel.lfr_n);
  int left_pad = (mel.lfr_m - 1) / 2;
  int left_pad_offset = (lfr_m - left_pad) * mel.n_mel;
  // Merge lfr_m frames as one,lfr_n frames per window
  T = T + (lfr_m - 1) / 2;
  std::vector<float> p;
  for (int i = 0; i < T_lrf; i++) {
    // the first frames need left padding
    if (i == 0) {
      // left padding
      for (int j = 0; j < left_pad; j++) {
        p.insert(p.end(), mel.data.begin(), mel.data.begin() + mel.n_mel);
      }
      p.insert(p.end(), mel.data.begin(), mel.data.begin() + left_pad_offset);
      out_feats.emplace_back(p);
      p.clear();
    } else {
      if (lfr_m <= T - i * lfr_n) {
        p.insert(p.end(), mel.data.begin() + (i * lfr_n - left_pad) * mel.n_mel,
                 mel.data.begin() + (i * lfr_n - left_pad + lfr_m) * mel.n_mel);
        out_feats.emplace_back(p);
        p.clear();
      } else {
        // Fill to lfr_m frames at last window if less than lfr_m frames  (copy
        // last frame)
        int num_padding = lfr_m - (T - i * lfr_n);
        for (int j = 0; j < (mel.n_len - i * lfr_n); j++) {
          p.insert(p.end(),
                   mel.data.begin() + (i * lfr_n - left_pad) * mel.n_mel,
                   mel.data.end());
        }
        for (int j = 0; j < num_padding; j++) {
          p.insert(p.end(), mel.data.end() - mel.n_mel, mel.data.end());
        }
        out_feats.emplace_back(p);
        p.clear();
      }
    }
  }

  // apply cvmn
  for (auto &out_feat : out_feats) {
    for (int j = 0; j < cmvn.cmvn_means.size(); j++) {
      out_feat[j] = (out_feat[j] + cmvn.cmvn_means[j]) * cmvn.cmvn_vars[j];
    }
  }

  return true;
}

bool load_wav_file(const char *filename, int32_t *sampling_rate,
                   std::vector<float> &data) {
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

    //        float scale = 32768;
    float scale = 1;
    for (int32_t i = 0; i != speech_len; ++i) {
      data[i] = (float)speech_buff[i] / scale;
    }
    return true;
  } else {
    return false;
  }
}
