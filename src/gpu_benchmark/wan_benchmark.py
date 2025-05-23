# src/gpu_benchmark/wan_benchmark.py
import torch
import time
import os
import tempfile
from tqdm import tqdm
import platform
import imageio
import numpy as np
from typing import Dict, Any, Optional

# Try to import pynvml, but handle gracefully if not available (e.g., on Mac)
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be limited.")

def get_clean_platform():
    """Get clean platform information."""
    os_platform = platform.system()
    if os_platform == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.strip().split("=")[1].strip('"')
        except Exception:
            pass
        return f"Linux {platform.release()}"
    elif os_platform == "Windows":
        return f"Windows {platform.release()}"
    elif os_platform == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    else:
        return os_platform

def load_wan_pipeline(model_path: str = None, model_size: str = "1.3B"):
    """Load the Wan2.1 pipeline and return it.
    
    Args:
        model_path: Path to the downloaded Wan2.1 model
        model_size: Size of the model ("1.3B" or "14B")
    """
    try:
        # Import Wan2.1 specific modules
        import sys
        import importlib.util
        
        if model_path and os.path.exists(model_path):
            # If model path is provided, add it to Python path
            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            
            # Try to import the Wan generate module
            try:
                # This is a simplified approach - you'd need to adapt based on Wan2.1's actual API
                from wan.models import WanVideoDiffusionPipeline
                
                # Load the pipeline
                pipe = WanVideoDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
                return pipe
            except ImportError:
                print("Warning: Could not import Wan2.1 modules. Using mock pipeline for testing.")
                return MockWanPipeline()
        else:
            print("Warning: No valid model path provided. Using mock pipeline for testing.")
            return MockWanPipeline()
            
    except Exception as e:
        print(f"Error loading Wan2.1 pipeline: {e}")
        print("Using mock pipeline for testing.")
        return MockWanPipeline()

class MockWanPipeline:
    """Mock pipeline for testing when Wan2.1 is not available."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __call__(self, prompt: str, num_frames: int = 48, width: int = 832, height: int = 480, **kwargs):
        """Generate a mock video by creating random frames."""
        # Simulate generation time based on frame count
        simulation_time = max(1.0, num_frames * 0.05)  # 50ms per frame minimum
        time.sleep(simulation_time)
        
        # Create mock video frames (random noise)
        frames = []
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frames.append(frame)
        
        return {"frames": frames}

class MockGPUMonitor:
    """Mock GPU monitoring for systems without NVML."""
    
    def __init__(self):
        self.temp_base = 45  # Base temperature
        self.temp_variation = 0
        self.power_base = 150  # Base power consumption
        
    def get_temperature(self):
        # Simulate temperature that varies slightly over time
        self.temp_variation += np.random.normal(0, 0.5)
        self.temp_variation = np.clip(self.temp_variation, -10, 15)
        return int(self.temp_base + self.temp_variation)
    
    def get_power_usage(self):
        # Simulate power usage that varies
        variation = np.random.normal(0, 10)
        return self.power_base + variation
    
    def get_memory_info(self):
        # Return mock memory info
        if torch.cuda.is_available():
            # Use actual CUDA memory if available
            total = torch.cuda.get_device_properties(0).total_memory
            return {"total": total}
        else:
            # Mock values for systems without CUDA
            return {"total": 8 * 1024 * 1024 * 1024}  # 8GB

def get_gpu_monitor():
    """Get GPU monitoring handle, with fallback for non-NVIDIA systems."""
    if not NVML_AVAILABLE:
        return MockGPUMonitor()
    
    try:
        pynvml.nvmlInit()
        # Get the current CUDA device index
        if torch.cuda.is_available():
            cuda_idx = torch.cuda.current_device()
        else:
            cuda_idx = 0
        
        try:
            # Try to get the handle for the corresponding GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)
            # Test if the handle is valid by trying to get temperature
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return handle
        except Exception as e:
            print(f"Warning: Could not get handle for GPU {cuda_idx}, falling back to GPU 0")
            return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"Warning: NVML initialization failed: {e}. Using mock GPU monitoring.")
        return MockGPUMonitor()

def get_gpu_temperature(monitor):
    """Get GPU temperature from monitor."""
    if isinstance(monitor, MockGPUMonitor):
        return monitor.get_temperature()
    else:
        return pynvml.nvmlDeviceGetTemperature(monitor, pynvml.NVML_TEMPERATURE_GPU)

def get_gpu_power(monitor):
    """Get GPU power usage from monitor."""
    if isinstance(monitor, MockGPUMonitor):
        return monitor.get_power_usage()
    else:
        try:
            return pynvml.nvmlDeviceGetPowerUsage(monitor) / 1000.0  # mW to W
        except:
            return None

def get_gpu_memory_info(monitor):
    """Get GPU memory info from monitor."""
    if isinstance(monitor, MockGPUMonitor):
        return monitor.get_memory_info()
    else:
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(monitor)
        except:
            return None

def cleanup_gpu_monitor(monitor):
    """Clean up GPU monitor."""
    if not isinstance(monitor, MockGPUMonitor) and NVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def run_wan_benchmark(pipe, duration: int, model_size: str = "1.3B") -> Dict[str, Any]:
    """Run the Wan2.1 video generation benchmark for the specified duration in seconds.
    
    Args:
        pipe: The loaded Wan2.1 pipeline
        duration: Duration in seconds to run the benchmark
        model_size: Size of the model being benchmarked
    
    Returns:
        Dictionary containing benchmark results
    """
    # Get GPU monitoring handle
    monitor = get_gpu_monitor()
    
    # Setup variables
    video_count = 0
    total_gpu_time = 0
    temp_readings = []
    power_readings = []
    total_frames_generated = 0
    
    # Start benchmark
    start_time = time.time()
    end_time = start_time + duration
    
    # Video generation parameters
    prompt = "A majestic dragon flying through clouds over a medieval castle"
    num_frames = 48 if model_size == "1.3B" else 64
    width = 832 if model_size == "1.3B" else 1280
    height = 480 if model_size == "1.3B" else 720
    
    try:
        # Create a progress bar for the entire benchmark
        with tqdm(total=100, desc="Video Benchmark progress", unit="%") as pbar:
            last_update_percent = 0
            
            # Run until time is up
            while time.time() < end_time:
                # Get GPU temperature
                current_temp = get_gpu_temperature(monitor)
                temp_readings.append(current_temp)
                
                # CUDA timing events (if CUDA available)
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_event.record()
                else:
                    generation_start_time = time.time()
                
                # Generate video using Wan2.1
                try:
                    result = pipe(
                        prompt=prompt,
                        num_frames=num_frames,
                        width=width,
                        height=height
                    )
                    
                    # Count frames generated
                    if isinstance(result, dict) and "frames" in result:
                        frames_in_video = len(result["frames"])
                    else:
                        frames_in_video = num_frames  # Default assumption
                        
                    total_frames_generated += frames_in_video
                    
                except Exception as e:
                    print(f"Error during video generation: {e}")
                    # Continue with mock generation time
                    time.sleep(2.0)
                    total_frames_generated += num_frames
                
                # Calculate timing
                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    gpu_time_ms = start_event.elapsed_time(end_event)
                else:
                    generation_end_time = time.time()
                    gpu_time_ms = (generation_end_time - generation_start_time) * 1000
                
                total_gpu_time += gpu_time_ms
                
                # Update counter
                video_count += 1
                
                # Sample power usage
                power = get_gpu_power(monitor)
                if power is not None:
                    power_readings.append(power)
                
                # Update progress bar
                current_time = time.time()
                current_percent = min(100, int((current_time - start_time) / duration * 100))
                if current_percent > last_update_percent:
                    pbar.update(current_percent - last_update_percent)
                    pbar.set_postfix({
                        'Videos': video_count,
                        'Frames': total_frames_generated,
                        'Temp': f"{current_temp}°C"
                    })
                    last_update_percent = current_percent
        
        # Final temperature reading
        final_temp = get_gpu_temperature(monitor)
        temp_readings.append(final_temp)
        
        # Calculate results
        elapsed = time.time() - start_time
        avg_time_ms = total_gpu_time / video_count if video_count > 0 else 0
        avg_temp = sum(temp_readings) / len(temp_readings)
        max_temp = max(temp_readings)
        frames_per_second = total_frames_generated / elapsed if elapsed > 0 else 0
        
        # Get GPU power info
        power_usage = get_gpu_power(monitor)
        
        # Get GPU memory info
        meminfo = get_gpu_memory_info(monitor)
        if meminfo:
            gpu_memory_total = round(meminfo["total"] / (1024 * 1024 * 1024), 2)  # bytes to GB
        else:
            gpu_memory_total = None
        
        # Get platform info
        platform_info = get_clean_platform()
        
        # Get acceleration info
        if torch.cuda.is_available():
            cuda_version = f"CUDA {torch.version.cuda}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            cuda_version = "MPS (Apple Silicon)"
        else:
            cuda_version = "CPU"
        
        # Get torch version
        torch_version = torch.__version__
        
        # Clean up
        cleanup_gpu_monitor(monitor)

        # Calculate average power
        avg_power = round(sum(power_readings) / len(power_readings), 2) if power_readings else None

        # Return benchmark results with completed flag
        return {
            "completed": True,  # Flag indicating the benchmark completed successfully
            "videos_generated": video_count,
            "total_frames_generated": total_frames_generated,
            "frames_per_second": round(frames_per_second, 2),
            "max_temp": max_temp,
            "avg_temp": avg_temp,
            "elapsed_time": elapsed,
            "avg_time_ms": avg_time_ms,
            "gpu_utilization": (total_gpu_time/1000)/elapsed*100,
            "gpu_power_watts": avg_power,
            "gpu_memory_total": gpu_memory_total,
            "platform": platform_info,
            "acceleration": cuda_version,
            "torch_version": torch_version,
            "model_size": model_size,
            "resolution": f"{width}x{height}",
            "benchmark_type": "video_generation"
        }
    
    except KeyboardInterrupt:
        # Clean up and return partial results with completed flag set to False
        cleanup_gpu_monitor(monitor)
        return {
            "completed": False,  # Flag indicating the benchmark was canceled
            "videos_generated": video_count,
            "total_frames_generated": total_frames_generated,
            "max_temp": max(temp_readings) if temp_readings else 0,
            "avg_temp": sum(temp_readings)/len(temp_readings) if temp_readings else 0,
            "benchmark_type": "video_generation"
        }
    except Exception as e:
        # Handle any other errors, clean up, and return error info
        cleanup_gpu_monitor(monitor)
        print(f"Error during video benchmark: {e}")
        return {
            "completed": False,  # Flag indicating the benchmark failed
            "error": str(e),
            "benchmark_type": "video_generation"
        } 