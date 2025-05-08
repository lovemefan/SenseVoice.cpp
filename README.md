# SenseVoice.cpp

「简体中文」|「[English](./README-EN.md)」

[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)是具有音频理解能力的音频基础模型， 
包括语音识别（ASR）、语种识别（LID）、语音情感识别（SER）和声学事件分类（AEC）或声学事件检测（AED）。
当前SenseVoice-small支持中、粤、英、日、韩语的多语言语音识别，情感识别和事件检测能力，具有极低的推理延迟。

本项目基于[ggml](https://github.com/ggerganov/ggml)推理框架。

## 1. 特性

1. 基于ggml，不依赖其他第三方库, 致力于端侧部署
2. 特征提取参考[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)库，支持多线程特征提取。
3. 支持flash attention解码
4. 支持Q3, Q4, Q5, Q6, Q8量化

### 1.1 backend支持

| 后端                                   | 平台                   | 是否支持 |
|--------------------------------------|----------------------|------|
| CPU                                  | All                  | ✅    |
| [Metal](./docs/build.md#metal-build) | Apple Silicon        | ✅    |   
| [BLAS](./docs/build.md#blas-build)   | All                  | ✅    |
| [CUDA](./docs/build.md#cuda)         | Nvidia GPU           | ✅    |
| [Vulkan](./docs/build.md#vulkan)     | GPU                  | ✅    |
| [Cann](./docs/build.md#cann)         | Ascend NPU           | 未测试  |
| [BLIS](./docs/backend/BLIS.md)       | All                  |      |
| [SYCL](./docs/backend/SYCL.md)       | Intel and Nvidia GPU |      |
| [MUSA](./docs/build.md#musa)         | Moore Threads GPU    |      |
| [hipBLAS](./docs/build.md#hipblas)   | AMD GPU              |      |


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

### 非流式语音识别 silero-vad + sense voice

#### 参数说明

以下列举的参数支持，未列举的暂不支持：
```bash
usage: ./bin/sense-voice-main [options] file.wav

options:
  -t N,      --threads N        [4     ] 解码使用的线程数
  -l LANG,   --language LANG    [auto  ] 语音代码 ('auto' 为自动检测), 支持 [`zh`, `en`, `yue`, `ja`, `ko`]，分别对应中文、英文、粤语、日语、韩语
  -m FNAME,  --model FNAME      [models/sense-voice-small-q4_k.gguf] gguf模型路径
  -f FNAME,  --file FNAME       [      ] wav文件路径， 当前仅支持16k采样率的音频
  --min_speech_duration_ms      [250   ] vad 参数， 切割音频最小长度，单位毫秒
  --max_speech_duration_ms      [15000 ] vad 参数， 切割音频最大长度，单位毫秒
  --min_silence_duration_ms     [100   ] vad 参数，静默最小长度
  -ng,       --no-gpu           [false ] 不使用GPU
  -fa,       --flash-attn       [false ] 使用flash attention 解码
  -itn,      --use-itn          [false ] 使用逆文本正则化，包括标点。
  -prfix,    --use-prefix       [false ] 输出语种、情感、事件、是否itn
 ```
#### 使用
```bash

git clone https://github.com/lovemefan/SenseVoice.cpp
cd SenseVoice.cpp
git submodule sync && git submodule update --init --recursive

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 8

# -t means thread num， -t 指定线程数
./bin/sense-voice-main -m /path/gguf-fp16-sense-voice-small.bin /path/asr_example_zh.wav  -t 4 -ng
```

### 输出

以下是使用sense-voice-q4_k模型在Macbook M1上输出:

```
$ ./bin/sense-voice-main -m /Users/Code/cpp-project/SenseVoice.cpp/scripts/resources/SenseVoiceGGUF/sense-voice-small-q4_k.gguf /Users/Downloads/asr_example_zh.wav  -t 1 -l auto -itn -prefix

sense_voice_small_init_from_file_with_params_no_state: loading model from '/Users/Code/cpp-project/SenseVoice.cpp/scripts/resources/SenseVoiceGGUF/sense-voice-small-q4_k.gguf'
sense_voice_init_with_params_no_state: use gpu    = 1
sense_voice_init_with_params_no_state: flash attn = 0
sense_voice_init_with_params_no_state: gpu_device = 0
sense_voice_init_with_params_no_state: devices    = 3
sense_voice_init_with_params_no_state: backends   = 3
sense_voice_model_load: version:      3
sense_voice_model_load: alignment:   32
sense_voice_model_load: data offset: 423680
sense_voice_model_load: loading model
sense_voice_model_load: n_vocab = 25055
sense_voice_model_load: n_encoder_hidden_state = 512
sense_voice_model_load: n_encoder_linear_units = 2048
sense_voice_model_load: n_encoder_attention_heads  = 4
sense_voice_model_load: n_encoder_layers = 50
sense_voice_model_load: n_mels  = 80
sense_voice_model_load: ftype  = 12
sense_voice_model_load: vocab[25055] loaded
sense_voice_default_buffer_type: using device Metal (Apple M1 Pro)
sense_voice_model_load: Metal total size =   181.86 MB
sense_voice_model_load: n_tensors: 1212
sense_voice_model_load: load SenseVoiceSmall takes 0.338000 second 
sense_voice_backend_init_gpu: using Metal backend
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M1 Pro
ggml_metal_init: picking default device: Apple M1 Pro
ggml_metal_init: using embedded metal library
ggml_metal_init: GPU name:   Apple M1 Pro
ggml_metal_init: GPU family: MTLGPUFamilyApple7  (1007)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
...
sense_voice_backend_init: using BLAS backend
sense_voice_backend_init: using CPU backend
sense_voice_init_state: kv pad  size  =    3.67 MB
sense_voice_init_state: compute buffer (encoder)   =    3.09 MB
sense_voice_init_state: compute buffer (encoder)   =   17.53 MB
sense_voice_init_state: compute buffer (decoder)   =    7.99 MB

system_info: n_threads = 1 / 8 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | COREML = 0 | OPENVINO = 0

main: processing audio (88747 samples, 5.54669 sec) , 1 threads, 1 processors, lang = auto...

[0.96-5.18] <|zh|><|NEUTRAL|><|Speech|><|withitn|>欢迎大家来体验达摩院推出的语音识别模型。

main: decoder audio use 0.103725 s, rtf is 0.018700. 
```
### 流式语音识别识别
流式的vad是基于信号处理实现的，区别于非流式的vad是使用模型实现的
```bash
usage: ./bin/sense-voice-stream [options]

options:
  -t N,     --threads N         [4      ] [SenseVoice] 解码使用的线程数
            --chunk_size        [100    ] vad chunk 大小(单位ms)
  -mmc      --min-mute-chunks   [10     ] 静音片段最小chunk数量
  -mnc      --max-nomute-chunks [80     ] 最大非静音chunk数量
            --use-vad           [false  ] 是否使用vad
            --use-prefix        [false  ] 是否使用 sensevoice的额外信息（语种、情感、事件、是否itn）
  -c ID,    --capture ID        [-1     ] [Device] capture device ID
  -l LANG,  --language LANG     [auto   ] [SenseVoice] 语音代码 ('auto' 为自动检测), 支持 [`zh`, `en`, `yue`, `ja`, `ko`]，分别对应中文、英文、粤语、日语、韩语
  -m FNAME, --model FNAME       [models/sense-voice-small-q4_k.gguf] [SenseVoice] 模型路径
  -ng,      --no-gpu           [false ] 不使用GPU
  -fa,      --flash-attn       [false ] 使用flash attention 解码
  -itn,     --use-itn          [false ] 使用逆文本正则化，包括标点。
 
```

```bash
sudo apt install libsdl2-dev
./bin/sense-voice-stream -m /path/gguf-fp16-sense-voice-small.bin
```

https://github.com/lovemefan/SenseVoice.cpp/releases/download/v1.4.0/sense-voice-straming.mp4

## 感谢以下项目

1. 本项目借用并模仿来自[whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp)
   的大部分c++代码
2. 参考来自funasr的paraformer模型结构以及前向计算 [FunASR](https://github.com/alibaba-damo-academy/FunASR)
3. 本项目参考并借用 [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)中的fbank特征提取算法。
   [FunASR](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/paraformer.cpp#L337C22-L372)
   中的lrf + cmvn 算法
4. 借用了大量的前期工作[paraformer.cpp](https://github.com/lovemefan/paraformer.cpp), paraformer.cpp项目后续将继续更新
