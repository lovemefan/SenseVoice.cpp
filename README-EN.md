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

#### Parameter Description
Only the following parameters are currently supported:
```bash
usage: ./bin/sense-voice-main [options] file.wav

options:
  -t N,      --threads N        [4     ] Number of decoding threads
  -l LANG,   --language LANG    [auto  ] Language code ('auto' for detection), supports [`zh`, `en`, `yue`, `ja`, `ko`]
  -m FNAME,  --model FNAME      [models/sense-voice-small-q4_k.gguf] Path to GGUF model
  -f FNAME,  --file FNAME       [      ] Path to WAV file (only supports 16kHz)
  --min_speech_duration_ms      [250   ] VAD parameter: minimum speech length in ms
  --max_speech_duration_ms      [15000 ] VAD parameter: maximum speech length in ms
  --min_silence_duration_ms     [100   ] VAD parameter: minimum silence length in ms
  -ng,       --no-gpu           [false ] Disable GPU
  -fa,       --flash-attn       [false ] Enable flash attention decoding
  -itn,      --use-itn          [false ] Use inverse text normalization (includes punctuation)
  -prfix,    --use-prefix       [false ] Output extra info: language, emotion, event, itn
```

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

Example output on MacBook M1 using the sense-voice-q4_k model:

```
$$ ./bin/sense-voice-main -m /Users/Code/cpp-project/SenseVoice.cpp/scripts/resources/SenseVoiceGGUF/sense-voice-small-q4_k.gguf /Users/Downloads/en.wav  -t 1 -l auto -itn -prefix

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

main: processing audio (114816 samples, 7.17600 sec) , 1 threads, 1 processors, lang = auto...

[1.12-3.42] <|en|><|NEUTRAL|><|Speech|><|withitn|>The tribal chief then called for the boy.
[3.87-6.53] <|en|><|NEUTRAL|><|Speech|><|withitn|>And presented him with 50 pieces of gold.

main: decoder audio use 0.135743 s, rtf is 0.018916. 

```

### Streaming Speech Recognition
```bash
sudo apt install libsdl2-dev
./bin/sense-voice-stream -m /path/gguf-fp16-sense-voice-small.bin
```
## Acknowledgements

1.	This project borrows and mimics most of the C++ code from [whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp).
2.	References the paraformer model structure and forward computation from [FunASR](https://github.com/alibaba-damo-academy/FunASR).
3.	Feature extraction algorithm borrowed from  [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) and the lrf + cmvn algorithm in [FunASR](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/paraformer.cpp#L337C22-L372).
4.	Utilizes a lot of preliminary work from [paraformer.cpp](https://github.com/lovemefan/paraformer.cpp), which will continue to be updated.

