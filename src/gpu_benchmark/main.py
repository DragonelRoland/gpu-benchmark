# src/gpu_benchmark/main.py
from .benchmark import load_pipeline, run_benchmark
from .wan_benchmark import load_wan_pipeline, run_wan_benchmark
from .database import upload_benchmark_results
import argparse
import torch

def main():
    """Entry point for the GPU benchmark command-line tool."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GPU Benchmark by United Compute")
    parser.add_argument("--provider", type=str, help="Cloud provider (e.g., RunPod, AWS, GCP) or Private", default="Private")
    parser.add_argument("--gpu", type=int, help="GPU device index to use (defaults to CUDA_VISIBLE_DEVICES or 0)", default=None)
    parser.add_argument("--benchmark_type", type=str, choices=["image", "video"], default="image", 
                       help="Type of benchmark: 'image' for Stable Diffusion, 'video' for Wan2.1")
    parser.add_argument("--model_id", type=str, help="Hugging Face model ID for Stable Diffusion (only for image benchmark)", 
                       default="yachty66/stable-diffusion-v1-5")
    parser.add_argument("--wan_model_path", type=str, help="Path to Wan2.1 model directory (only for video benchmark)", default=None)
    parser.add_argument("--wan_model_size", type=str, choices=["1.3B", "14B"], default="1.3B",
                       help="Size of Wan2.1 model to use (only for video benchmark)")
    args = parser.parse_args()
    
    # If GPU device is specified, set it
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    # Convert provider to lowercase
    provider = args.provider.lower()
    
    # Simple start message
    print("GPU Benchmark starting...")
    print("This benchmark will run for 5 minutes")
    
    # Fixed duration
    duration = 300  # 300 seconds
    
    if args.benchmark_type == "image":
        # Run Stable Diffusion image generation benchmark
        print(f"Loading Stable Diffusion pipeline: {args.model_id}...")
        pipe = load_pipeline(model_id=args.model_id)
        print("Pipeline loaded successfully!")
        
        print("Running image generation benchmark...")
        results = run_benchmark(pipe=pipe, duration=duration)
        
        # Display results
        if results.get("completed", False):
            print("\n" + "="*50)
            print("IMAGE GENERATION BENCHMARK RESULTS:")
            print(f"Images Generated: {results['images_generated']}")
            print(f"Max GPU Temperature: {results['max_temp']}°C")
            print(f"Avg GPU Temperature: {results['avg_temp']:.1f}°C")
            if results.get('gpu_power_watts'):
                print(f"GPU Power: {results['gpu_power_watts']}W")
            if results.get('gpu_memory_total'):
                print(f"GPU Memory: {results['gpu_memory_total']}GB")
            if results.get('platform'):
                print(f"Platform: {results['platform']}")
            if results.get('acceleration'):
                print(f"Acceleration: {results['acceleration']}")
            if results.get('torch_version'):
                print(f"PyTorch Version: {results['torch_version']}")
            print(f"Provider: {provider}")
            print(f"Model ID: {args.model_id}")
            print("="*50)
            
            print("\nSubmitting to benchmark results...")
            upload_benchmark_results(
                image_count=results['images_generated'],
                max_temp=results['max_temp'],
                avg_temp=results['avg_temp'],
                gpu_power_watts=results.get('gpu_power_watts'),
                gpu_memory_total=results.get('gpu_memory_total'),
                platform=results.get('platform'),
                acceleration=results.get('acceleration'),
                torch_version=results.get('torch_version'),
                cloud_provider=provider,
                model_id=args.model_id,
                benchmark_type="image_generation"
            )
        else:
            print("\nImage benchmark was canceled or failed. Results not submitted.")
    
    elif args.benchmark_type == "video":
        # Run Wan2.1 video generation benchmark
        print(f"Loading Wan2.1 pipeline (Model: {args.wan_model_size})...")
        if args.wan_model_path:
            print(f"Model path: {args.wan_model_path}")
        else:
            print("No model path provided, using mock pipeline for testing")
        
        pipe = load_wan_pipeline(model_path=args.wan_model_path, model_size=args.wan_model_size)
        print("Pipeline loaded successfully!")
        
        print("Running video generation benchmark...")
        results = run_wan_benchmark(pipe=pipe, duration=duration, model_size=args.wan_model_size)
        
        # Display results
        if results.get("completed", False):
            print("\n" + "="*50)
            print("VIDEO GENERATION BENCHMARK RESULTS:")
            print(f"Videos Generated: {results['videos_generated']}")
            print(f"Total Frames Generated: {results['total_frames_generated']}")
            print(f"Frames Per Second: {results['frames_per_second']}")
            print(f"Max GPU Temperature: {results['max_temp']}°C")
            print(f"Avg GPU Temperature: {results['avg_temp']:.1f}°C")
            if results.get('gpu_power_watts'):
                print(f"GPU Power: {results['gpu_power_watts']}W")
            if results.get('gpu_memory_total'):
                print(f"GPU Memory: {results['gpu_memory_total']}GB")
            if results.get('platform'):
                print(f"Platform: {results['platform']}")
            if results.get('acceleration'):
                print(f"Acceleration: {results['acceleration']}")
            if results.get('torch_version'):
                print(f"PyTorch Version: {results['torch_version']}")
            print(f"Provider: {provider}")
            print(f"Model Size: {args.wan_model_size}")
            print(f"Resolution: {results.get('resolution', 'Unknown')}")
            print("="*50)
            
            print("\nSubmitting to benchmark results...")
            upload_benchmark_results(
                video_count=results['videos_generated'],
                total_frames=results['total_frames_generated'],
                frames_per_second=results['frames_per_second'],
                max_temp=results['max_temp'],
                avg_temp=results['avg_temp'],
                gpu_power_watts=results.get('gpu_power_watts'),
                gpu_memory_total=results.get('gpu_memory_total'),
                platform=results.get('platform'),
                acceleration=results.get('acceleration'),
                torch_version=results.get('torch_version'),
                cloud_provider=provider,
                model_size=args.wan_model_size,
                resolution=results.get('resolution'),
                benchmark_type="video_generation"
            )
        else:
            print("\nVideo benchmark was canceled or failed. Results not submitted.")
    
    print("Benchmark completed")

if __name__ == "__main__":
    main()