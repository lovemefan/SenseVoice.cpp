# build SenseVoice.cpp locally

To get the code 

```bash
git clone https://github.com/lovemefan/SenseVoice.cpp
cd SenseVoice.cpp
git submodule sync && git submodule update --init --recursive
```

## cpu-build

Use cmake

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 8
```

## blas-build

Building the program with BLAS support may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). Support with CPU-only BLAS implementations doesn't affect the normal generation performance. We may see generation performance improvements with GPU-involved BLAS implementations, e.g. cuBLAS, hipBLAS. There are currently several different BLAS implementations available for build and use:

### Accelerate Framework:

This is only available on Mac PCs and it's enabled by default. You can just build using the normal instructions.

### OpenBLAS:

This provides BLAS acceleration using only the CPU. Make sure to have OpenBLAS installed on your machine.



Using CMake on Linux:
```bash
mkdir build && cd build
cmake -DGGML_BLAS_VENDOR=OpenBLAS .. && make -j 8
```

## metal-build
On MacOS, Metal is enabled by default. 
Using Metal makes the computation run on the GPU. 
To disable the Metal build at compile time use the GGML_NO_METAL=1 flag or the GGML_METAL=OFF cmake option.

## cuda

This provides GPU acceleration using the CUDA cores of your Nvidia GPU. Make sure to have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

For Jetson user, if you have Jetson Orin, you can try this: [Offical Support](https://www.jetson-ai-lab.com/tutorial_text-generation.html). If you are using an old model(nano/TX2), need some additional operations before compiling.

- Using `CMake`:

  ```bash
  mkdir build && cd build
  cmake -DGGML_CUDA=ON .. && make -j 8
  ```

The environment variable [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) can be used to specify which GPU(s) will be used.

The environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` can be used to enable unified memory in Linux. This allows swapping to system RAM instead of crashing when the GPU VRAM is exhausted. In Windows this setting is available in the NVIDIA control panel as `System Memory Fallback`.

The following compilation options are also available to tweak performance:

| Option                        | Legal values           | Default | Description                                                                                                                                                                                                                                                                             |
|-------------------------------|------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GGML_CUDA_FORCE_DMMV          | Boolean                | false   | Force the use of dequantization + matrix vector multiplication kernels instead of using kernels that do matrix vector multiplication on quantized data. By default the decision is made based on compute capability (MMVQ for 6.1/Pascal/GTX 1000 or higher). Does not affect k-quants. |
| GGML_CUDA_DMMV_X              | Positive integer >= 32 | 32      | Number of values in x direction processed by the CUDA dequantization + matrix vector multiplication kernel per iteration. Increasing this value can improve performance on fast GPUs. Power of 2 heavily recommended. Does not affect k-quants.                                         |
| GGML_CUDA_MMV_Y               | Positive integer       | 1       | Block size in y direction for the CUDA mul mat vec kernels. Increasing this value can improve performance on fast GPUs. Power of 2 recommended.                                                                                                                                         |
| GGML_CUDA_FORCE_MMQ           | Boolean                | false   | Force the use of custom matrix multiplication kernels for quantized models instead of FP16 cuBLAS even if there is no int8 tensor core implementation available (affects V100, RDNA3). MMQ kernels are enabled by default on GPUs with int8 tensor core support. With MMQ force enabled, speed for large batch sizes will be worse but VRAM consumption will be lower.                       |
| GGML_CUDA_FORCE_CUBLAS        | Boolean                | false   | Force the use of FP16 cuBLAS instead of custom matrix multiplication kernels for quantized models                                                                                                                                                                                       |
| GGML_CUDA_F16                 | Boolean                | false   | If enabled, use half-precision floating point arithmetic for the CUDA dequantization + mul mat vec kernels and for the q4_1 and q5_1 matrix matrix multiplication kernels. Can improve performance on relatively recent GPUs.                                                           |
| GGML_CUDA_KQUANTS_ITER        | 1 or 2                 | 2       | Number of values processed per iteration and per CUDA thread for Q2_K and Q6_K quantization formats. Setting this value to 1 can improve performance for slow GPUs.                                                                                                                     |
| GGML_CUDA_PEER_MAX_BATCH_SIZE | Positive integer       | 128     | Maximum batch size for which to enable peer access between multiple GPUs. Peer access requires either Linux or NVLink. When using NVLink enabling peer access for larger batch sizes is potentially beneficial.                                                                         |
| GGML_CUDA_FA_ALL_QUANTS       | Boolean                | false   | Compile support for all KV cache quantization type (combinations) for the FlashAttention CUDA kernels. More fine-grained control over KV cache size but compilation takes much longer.                                                                                                  |


## vulkan
**Windows**

#### w64devkit

Download and extract [w64devkit](https://github.com/skeeto/w64devkit/releases).

Download and install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows). When selecting components, only the Vulkan SDK Core is required.

Launch `w64devkit.exe` and run the following commands to copy Vulkan dependencies:
```sh
SDK_VERSION=1.3.283.0
cp /VulkanSDK/$SDK_VERSION/Bin/glslc.exe $W64DEVKIT_HOME/bin/
cp /VulkanSDK/$SDK_VERSION/Lib/vulkan-1.lib $W64DEVKIT_HOME/x86_64-w64-mingw32/lib/
cp -r /VulkanSDK/$SDK_VERSION/Include/* $W64DEVKIT_HOME/x86_64-w64-mingw32/include/
cat > $W64DEVKIT_HOME/x86_64-w64-mingw32/lib/pkgconfig/vulkan.pc <<EOF
Name: Vulkan-Loader
Description: Vulkan Loader
Version: $SDK_VERSION
Libs: -lvulkan-1
EOF

```

#### MSYS2
Install [MSYS2](https://www.msys2.org/) and then run the following commands in a UCRT terminal to install dependencies.
  ```sh
  pacman -S git \
      mingw-w64-ucrt-x86_64-gcc \
      mingw-w64-ucrt-x86_64-cmake \
      mingw-w64-ucrt-x86_64-vulkan-devel \
      mingw-w64-ucrt-x86_64-shaderc
  ```

```sh
mkdir build && cd build
cmake -DGGML_VULKAN=ON .. && make -j 8
```

**With docker**:

You don't need to install Vulkan SDK. It will be installed inside the container.

here is the dockerfile

```sh
# Build the image
docker build -t sense-voice-cpp-vulkan -f .devops/sense-voice-cli-vulkan.Dockerfile .

# Then, use it:
docker run -it --rm -v "$(pwd):/app"  sense-voice-cpp-vulkan /app/build/bin/sense-voice-main -m "/app/models/YOUR_MODEL_FILE" -t 8 -l auto "YOUR WAV FILE"
```

**Without docker**:

Firstly, you need to make sure you have installed [Vulkan SDK](https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html)

For example, on Ubuntu 22.04 (jammy), use the command below:

```bash
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
apt update -y
apt-get install -y vulkan-sdk
# To verify the installation, use the command below:
vulkaninfo
```

Alternatively your package manager might be able to provide the appropriate libraries.
For example for Ubuntu 22.04 you can install `libvulkan-dev` instead.
For Fedora 40, you can install `vulkan-devel`, `glslc` and `glslang` packages.

Then, build SenseVoice.cpp using the cmake command below:

```bash
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release
# Test the output binary (with "-ngl 33" to offload all layers to GPU)
./build/bin/sense-voice-main -m "/app/models/YOUR_MODEL_FILE" -t 8 -l auto "YOUR WAV FILE"

# You should see in the output, ggml_vulkan detected your GPU. For example:
# ggml_vulkan: Using Intel(R) Graphics (ADL GT2) | uma: 1 | fp16: 1 | warp size: 32
```

### CANN
This provides NPU acceleration using the AI cores of your Ascend NPU. And [CANN](https://www.hiascend.com/en/software/cann) is a hierarchical APIs to help you to quickly build AI applications and service based on Ascend NPU.

For more information about Ascend NPU in [Ascend Community](https://www.hiascend.com/en/).

Make sure to have the CANN toolkit installed. You can download it from here: [CANN Toolkit](https://www.hiascend.com/developer/download/community/result?module=cann)

Go to `sensevoice.cpp` directory and build using CMake.
```bash
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
cmake --build build --config release
```

You can test with:

`./build/bin/sense-voice-main -m "/app/models/YOUR_MODEL_FILE" -t 8 -l auto "YOUR WAV FILE"`

