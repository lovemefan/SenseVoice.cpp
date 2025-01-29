# SenseVoice.cpp

[「简体中文」](./README.md)|「English」


[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)SenseVoice is an audio foundation model with audio understanding capabilities, 
including Automatic Speech Recognition (ASR), Language Identification (LID), Speech Emotion Recognition (SER), 
and Acoustic Event Classification (AEC) or Acoustic Event Detection (AED). 
Currently, SenseVoice-small supports multilingual speech recognition, emotion recognition, and event detection capabilities in 
Mandarin, Cantonese, English, Japanese, and Korean, with extremely low inference latency.

This project is based on the [ggml](https://github.com/ggerganov/ggml) framework.

## 1.  Features

1.	Based on ggml, it does not rely on other third-party libraries and is committed to edge deployment.
2.	Feature extraction references the [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) library, supporting multi-threaded feature extraction.
3.	Support Flash attention decoding  
4.  Support Q3, Q4, Q5, Q6, Q8 quantization.

### 1.1 Backend Support


1.	Support more backends. In theory, ggml supports the following backends, and future adaptations will be gradually made. Contributions are welcome.

| Backend                                   | Device               | Supported    |
|--------------------------------------|----------------------|--------------|
| CPU                                  | All                  | ✅            |
| [Metal](./docs/build.md#metal-build) | Apple Silicon        | ✅            |   
| [BLAS](./docs/build.md#blas-build)   | All                  | ✅            |
| [CUDA](./docs/build.md#cuda)         | Nvidia GPU           | ✅            |
| [Vulkan](./docs/build.md#vulkan)     | GPU                  | ✅            |
| [Cann](./docs/build.md#cann)         | Ascend NPU           | Not yet tested |
| [BLIS](./docs/backend/BLIS.md)       | All                  |              |
| [SYCL](./docs/backend/SYCL.md)       | Intel and Nvidia GPU |              |
| [MUSA](./docs/build.md#musa)         | Moore Threads GPU    |              |
| [hipBLAS](./docs/build.md#hipblas)   | AMD GPU              |              |


## 2. Usage

### Download Model or Convert Model
You can download the model directly from the links below:

[huggingface](https://huggingface.co/lovemefan/sense-voice-gguf)
[modelscope](https://www.modelscope.cn/models/lovemefan/SenseVoiceGGUF)

```bash
git lfs install
git clone https://huggingface.co/lovemefan/sense-voice-gguf.git
# 或从modelscope下载
git clone https://www.modelscope.cn/models/lovemefan/SenseVoiceGGUF.git
```

Alternatively, download the official model and convert it yourself:
```bash
# Download the official model
git lfs install
git clone https://www.modelscope.cn/iic/SenseVoiceSmall.git
# Convert the model
python scripts/convert-pt-to-gguf.py \
--model SenseVoiceSmall \
--output /path/to/export/gguf-fp32-sense-voice-small.bin \
--out_type f32
```

### Non-Streaming Speech Recognition (Silero-VAD + SenseVoice)
```bash

git clone https://github.com/lovemefan/SenseVoice.cpp
cd SenseVoice.cpp
git submodule sync && git submodule update --init --recursive

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 8

# -t means thread num
./bin/sense-voice-main -m /path/gguf-fp16-sense-voice-small.bin /path/asr_example_zh.wav  -t 4 -ng
```

### Output

Currently using the sense-voice-f16 model for output:

```
$./bin/sense-voice-main -m /data/code/SenseVoice.cpp/scripts/resources/gguf-fp16-sense-voice.bin /data/code/SenseVoice.cpp/scripts/resources/SenseVoiceSmall/example/asr_example_zh.wav  -t 4

sense_voice_small_init_from_file_with_params_no_state: loading model from '/data/code/SenseVoice.cpp/scripts/resources/gguf-fp16-sense-voice-small.bin'     
sense_voice_model_load: version:      3                                                                                                                     
sense_voice_model_load: alignment:   32 
sense_voice_model_load: data offset: 444480                                                                                                     
sense_voice_model_load: loading model                                                                                                                       
sense_voice_model_load: n_vocab = 25055                                                                                                                     
sense_voice_model_load: n_encoder_hidden_state = 512                                                                                                        
sense_voice_model_load: n_encoder_linear_units = 2048                                                                                                       
sense_voice_model_load: n_encoder_attention_heads  = 4                                                                                                      
sense_voice_model_load: n_encoder_layers = 50                                                                                                               
sense_voice_model_load: n_mels  = 80                                                                                                                        
sense_voice_model_load: ftype  = 1                                                                                                                          
sense_voice_model_load: vocab[25055] loaded 
sense_voice_model_load: CPU total size =   468.98 MB
sense_voice_model_load: n_tensors: 1197
sense_voice_model_load: load SenseVoiceSmall takes 0.213000 second 
sense_voice_init_state: compute buffer (encoder)   =   50.40 MB
sense_voice_init_state: compute buffer (decoder)   =   13.72 MB

system_info: n_threads = 4 / 256 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | METAL = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | CUDA = 0 | COREML = 0 | OPENVINO = 0

main: processing audio (88747 samples, 5.54669 sec) , 4 threads, 1 processors, lang = auto...

sense_voice_pcm_to_feature_with_state: calculate fbank and cmvn takes 7.207 ms
<|zh|><|NEUTRAL|><|Speech|><|withitn|>欢迎大家来体验达摩院推出的语音识别模型。
sense_voice_full_with_state: decoder audio use 1.011289 s, rtf is 0.182323.
```

### Streaming Speech Recognition
```bash
./bin/sense-voice-stream -m /path/gguf-fp16-sense-voice-small.bin
```
## Acknowledgements

1.	This project borrows and mimics most of the C++ code from [whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp).
2.	References the paraformer model structure and forward computation from [FunASR](https://github.com/alibaba-damo-academy/FunASR).
3.	Feature extraction algorithm borrowed from  [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) and the lrf + cmvn algorithm in [FunASR](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/paraformer.cpp#L337C22-L372).
4.	Utilizes a lot of preliminary work from [paraformer.cpp](https://github.com/lovemefan/paraformer.cpp), which will continue to be updated.

