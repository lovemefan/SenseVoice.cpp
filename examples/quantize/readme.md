```bash
./build/bin/sense-voice-quantize "input gguf model file" "out put model file" type

#for example 
./build/bin/sense-voice-quantize sense-voice-small-fp32.gguf sense-voice-small-q4_0.bin q4_0
```

support type for now

|quantize type| ggml type              |
|---|------------------------|
|q4_0| GGML_FTYPE_MOSTLY_Q4_0 |
|q4_1| GGML_FTYPE_MOSTLY_Q4_1 |
|q5_0| GGML_FTYPE_MOSTLY_Q5_0 |
|q5_1| GGML_FTYPE_MOSTLY_Q5_1 |
|q8_0| GGML_FTYPE_MOSTLY_Q8_0 |
|q2_k| GGML_FTYPE_MOSTLY_Q2_K |
|q3_k| GGML_FTYPE_MOSTLY_Q3_K |
|q4_k| GGML_FTYPE_MOSTLY_Q4_K |
|q5_k| GGML_FTYPE_MOSTLY_Q5_K |
|q6_k| GGML_FTYPE_MOSTLY_Q6_K |