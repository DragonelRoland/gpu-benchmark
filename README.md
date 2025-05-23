# GPU Benchmark by [United Compute](https://www.unitedcompute.ai)

A simple CLI tool to benchmark your GPU's performance with Stable Diffusion and compare results in our global benchmark results.

![United Compute Logo](https://www.unitedcompute.ai/logo.png)

## Installation

```bash
pip install gpu-benchmark
```

## Usage

Run the benchmark (takes 5 minutes after the pipeline is loaded):

```bash
gpu-benchmark
```

### Optional Arguments

If you're running on a cloud provider, specify it with the `--provider` flag, e.g. Runpod:
```bash
gpu-benchmark --provider runpod
```

For multi-GPU systems, you can select a specific GPU in two ways:

1. Using the `--gpu` flag:
```bash
gpu-benchmark --gpu 1  # Uses GPU index 1
```

2. Using CUDA_VISIBLE_DEVICES:
```bash
CUDA_VISIBLE_DEVICES=1 gpu-benchmark  # Uses GPU index 1
```

The tool will:

1. Load a Stable Diffusion pipeline
2. Generate images for 5 minutes
3. Count image generations and track GPU temperature
4. Upload results to the [United Compute Benchmark Results](https://www.unitedcompute.ai/gpu-benchmark)

## What it measures

- **Images Generated**: Number of Stable Diffusion images generated in 5 minutes
- **Max Heat**: Maximum GPU temperature reached (°C)
- **Avg Heat**: Average GPU temperature during the benchmark (°C)
- **Country**: Your location (detected automatically)
- **GPU Power**: Power consumption in watts (W)
- **GPU Memory**: Total GPU memory in gigabytes (GB)
- **Platform**: Operating system information
- **Acceleration**: CUDA version
- **PyTorch Version**: PyTorch library version

## Requirements

- CUDA-compatible NVIDIA GPU
- Minimum 4GB VRAM (benchmark uses ~3.3GB VRAM in fp16 mode)
- Python 3.8+
- Internet connection (for results submission - although you can run the test offline too)

## Links

- [Official Website](https://www.unitedcompute.ai)
- [GPU Benchmark Results](https://www.unitedcompute.ai/gpu-benchmark)
