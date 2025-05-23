# Wan2.1 GPU Benchmark Setup Guide

This guide shows you how to set up and use Wan2.1 video generation with your GPU benchmark tool.

## Installation

### Step 1: Install the Updated Benchmark Tool

```bash
# Install the updated benchmark with Wan2.1 support
pip install -e .
```

### Step 2: Install Wan2.1 Dependencies

The main dependencies are already included in `pyproject.toml`, but you may need to install `flash_attn` separately:

```bash
# For most systems
pip install flash-attn --no-build-isolation

# Alternative: install wheel first
pip install wheel
pip install flash-attn
```

### Step 3: Download Wan2.1 Models (Optional)

For a real Wan2.1 benchmark, you need to download the actual models:

**Using Hugging Face CLI:**
```bash
pip install "huggingface_hub[cli]"

# Download 1.3B model (requires ~8GB VRAM)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

# Download 14B model (requires ~24GB VRAM)  
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

**Using ModelScope CLI:**
```bash
pip install modelscope

# Download 1.3B model
modelscope download Wan-AI/Wan2.1-T2V-1.3B --local_dir ./Wan2.1-T2V-1.3B

# Download 14B model
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B
```

## Usage

### Image Generation Benchmark (Stable Diffusion)

```bash
# Run the traditional image generation benchmark
python -m gpu_benchmark.main --benchmark_type image

# With custom model
python -m gpu_benchmark.main --benchmark_type image --model_id "runwayml/stable-diffusion-v1-5"
```

### Video Generation Benchmark (Wan2.1)

```bash
# Run with mock pipeline (for testing without downloading models)
python -m gpu_benchmark.main --benchmark_type video --wan_model_size 1.3B

# Run with real Wan2.1 model (after downloading)
python -m gpu_benchmark.main --benchmark_type video --wan_model_path ./Wan2.1-T2V-1.3B --wan_model_size 1.3B

# Run with 14B model (requires high-end GPU)
python -m gpu_benchmark.main --benchmark_type video --wan_model_path ./Wan2.1-T2V-14B --wan_model_size 14B
```

### Additional Options

```bash
# Specify GPU device
python -m gpu_benchmark.main --benchmark_type video --gpu 0

# Specify cloud provider
python -m gpu_benchmark.main --benchmark_type video --provider "AWS" --wan_model_size 1.3B
```

## System Requirements

### Minimum Requirements (Mock Pipeline)
- **GPU**: Any CUDA-compatible GPU with 4GB+ VRAM
- **RAM**: 8GB system RAM
- **Storage**: 2GB for dependencies

### Real Wan2.1 Requirements

**For 1.3B Model:**
- **GPU**: RTX 3080/3090 or better with 8GB+ VRAM
- **RAM**: 16GB system RAM  
- **Storage**: 15GB for model and cache

**For 14B Model:**
- **GPU**: RTX 4090, A100, or H100 with 24GB+ VRAM
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB for model and cache

## Benchmark Metrics

The video benchmark tracks:

- **Videos Generated**: Number of complete videos created
- **Total Frames Generated**: Total frames across all videos
- **Frames Per Second**: Average generation rate
- **GPU Temperature**: Max and average during generation
- **GPU Power Usage**: Power consumption during generation
- **Model Performance**: Resolution and timing metrics

## Troubleshooting

### Common Issues

**1. `flash_attn` Installation Fails**
```bash
# Try with no build isolation
pip install flash-attn --no-build-isolation

# Or install build tools first
pip install wheel build cmake ninja
```

**2. CUDA Out of Memory**
```bash
# Use smaller model
python -m gpu_benchmark.main --benchmark_type video --wan_model_size 1.3B

# Or test with mock pipeline first
python -m gpu_benchmark.main --benchmark_type video
```

**3. Model Download Issues**
- Ensure you have sufficient disk space
- Check your internet connection
- Try using ModelScope instead of Hugging Face

### Performance Tips

1. **Use the mock pipeline first** to test your setup before downloading large models
2. **Close other GPU applications** before running benchmarks
3. **Monitor GPU temperature** - the benchmark will stop if overheating occurs
4. **Use SSD storage** for better model loading performance

## Understanding Results

### Video Generation vs Image Generation

- **Video benchmarks** measure sustained performance over longer generation times
- **Frame rates** indicate how efficiently your GPU handles temporal consistency
- **Memory usage** is typically higher for video models
- **Temperature** tends to be higher due to longer continuous GPU usage

### Interpreting Metrics

- **High frames/second**: Your GPU handles video generation efficiently  
- **Low video count but high total frames**: Model generates longer, higher-quality videos
- **High power usage**: Normal for video generation - it's more demanding than images

## Next Steps

1. **Start with mock pipeline** to verify setup
2. **Download 1.3B model** if you have 8GB+ VRAM
3. **Compare with image benchmarks** to see the difference
4. **Upgrade to 14B model** for ultimate performance testing (requires high-end hardware)

The tool now supports both traditional image generation benchmarking and cutting-edge video generation testing with Wan2.1! 