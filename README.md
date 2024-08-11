# SenseVoice.cpp

「简体中文」|「[English](./README-EN.md)」

[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)是具有音频理解能力的音频基础模型， 
包括语音识别（ASR）、语种识别（LID）、语音情感识别（SER）和声学事件分类（AEC）或声学事件检测（AED）。
当前SenseVoice-small支持中、粤、英、日、韩语的多语言语音识别，情感识别和事件检测能力，具有极低的推理延迟。

本项目基于[ggml](https://github.com/ggerganov/ggml)推理框架。

## 1. 特性

1. 基于ggml，不依赖其他第三方库, 致力于端侧部署
2. 特征提取参考[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)库，支持多线程特征提取。
3. 可以使用flash attention解码（速度没有明显提升🤔不知道为啥）

### 1.1 未来计划

1. 支持更多backend , 理论上来说，ggml支持以下后端，后续将会慢慢适配，欢迎贡献。

| 后端                                   | 平台                   | 是否支持 |
|--------------------------------------|----------------------|------|
| CPU                                  | All                  | ✅    |
| [Metal](./docs/build.md#metal-build) | Apple Silicon        |      |   
| [BLAS](./docs/build.md#blas-build)   | All                  |      |
| [BLIS](./docs/backend/BLIS.md)       | All                  |      |
| [SYCL](./docs/backend/SYCL.md)       | Intel and Nvidia GPU |      |
| [MUSA](./docs/build.md#musa)         | Moore Threads GPU    |      |
| [CUDA](./docs/build.md#cuda)         | Nvidia GPU           |      |
| [hipBLAS](./docs/build.md#hipblas)   | AMD GPU              |      |
| [Vulkan](./docs/build.md#vulkan)     | GPU                  |      |
| [Cann](./docs/build.md#vulkan)       | Ascend NPU           |      |


2. 支持更多量化模型
3. 提升性能
4. 修bug

## 2. 使用

### 直接下载模型或转换模型
可以直接从下面链接下载模型

[huggingface](https://huggingface.co/lovemefan/sense-voice-gguf)
[modelscope](https://www.modelscope.cn/models/lovemefan/SenseVoiceGGUF)

```bash
git lfs install
git clone https://huggingface.co/lovemefan/sense-voice-gguf.git
# 或从modelscope下载
git clone https://www.modelscope.cn/models/lovemefan/SenseVoiceGGUF.git
```

或许自行下载官方模型转换
```bash
# 下载官方模型
git lfs install
git clone https://www.modelscope.cn/iic/SenseVoiceSmall.git
# 转换模型
python scripts/convert-pt-to-gguf.py \
--model SenseVoiceSmall \
--output /path/to/export/gguf-fp32-sense-voice-small.bin \
--out_type f32
```

### RUN
```bash

git clone https://github.com/lovemefan/SenseVoice.cpp
cd SenseVoice.cpp
git submodule sync && git submodule update --init --recursive

mkdir build && cd build
cmake .. && make -j 8

# -t means thread num， -t 指定线程数
./bin/sense-voice-main -m /path/gguf-fp16-sense-voice-small.bin /path/asr_example_zh.wav  -t 4
```

### 输出

当前使用sense-voice-f16模型输出

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
## 感谢以下项目

1. 本项目借用并模仿来自[whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp)
   的大部分c++代码
2. 参考来自funasr的paraformer模型结构以及前向计算 [FunASR](https://github.com/alibaba-damo-academy/FunASR)
3. 本项目参考并借用 [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)中的fbank特征提取算法。
   [FunASR](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/paraformer.cpp#L337C22-L372)
   中的lrf + cmvn 算法
4. 借用了大量的前期工作[paraformer.cpp](https://github.com/lovemefan/paraformer.cpp), paraformer.cpp项目后续将继续更新
