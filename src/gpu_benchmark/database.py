# src/gpu_benchmark/database.py
import requests
import datetime
import torch
import platform

# Hardcoded Supabase credentials (anon key is designed to be public)
SUPABASE_URL = "https://jftqjabhnesfphpkoilc.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmdHFqYWJobmVzZnBocGtvaWxjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ5NzI4NzIsImV4cCI6MjA2MDU0ODg3Mn0.S0ZdRIauUyMhdVJtYFNquvnlW3dV1wxERy7YrurZyag"

def get_gpu_type():
    """Get GPU type information, handling different systems."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For Apple Silicon Macs
            system_info = platform.uname()
            if 'arm64' in system_info.machine.lower():
                return f"Apple Silicon ({system_info.machine})"
            else:
                return "Apple MPS"
        else:
            return "CPU (No GPU acceleration)"
    except Exception as e:
        return f"Unknown GPU ({str(e)})"

def country_code_to_flag(country_code):
    """Convert country code to flag emoji."""
    if len(country_code) != 2 or not country_code.isalpha():
        return "🏳️"  # White flag for unknown
    
    # Convert each letter to regional indicator symbol
    # A-Z: 0x41-0x5A -> regional indicators: 0x1F1E6-0x1F1FF
    return ''.join(chr(ord(c.upper()) - ord('A') + ord('🇦')) for c in country_code)

def get_country_flag():
    """Get country flag emoji based on IP."""
    try:
        country_response = requests.get("https://ipinfo.io/json")
        country_code = country_response.json().get("country", "Unknown")
        return country_code_to_flag(country_code)
    except Exception as e:
        print(f"Error getting country info: {e}")
        return "🏳️"  # White flag for unknown

def upload_benchmark_results(cloud_provider="Private", benchmark_type="image_generation", **kwargs):
    """Upload benchmark results to Supabase database.
    
    Args:
        cloud_provider: Cloud provider name (default: "Private")
        benchmark_type: Type of benchmark ("image_generation" or "video_generation")
        **kwargs: Additional fields to upload
        
    Returns:
        tuple: (success, message, record_id)
    """
    # Get country flag
    flag_emoji = get_country_flag()
    
    # Prepare benchmark results
    benchmark_results = {
        "created_at": datetime.datetime.now().isoformat(),
        "gpu_type": get_gpu_type(),
        "country": flag_emoji,
        "provider": cloud_provider,
        "benchmark_type": benchmark_type
    }
    
    # Handle different benchmark types
    if benchmark_type == "image_generation":
        # For image generation benchmarks (Stable Diffusion)
        required_fields = ["image_count", "max_temp", "avg_temp"]
        for field in required_fields:
            if field in kwargs:
                if field == "image_count":
                    benchmark_results["number_images_generated"] = kwargs[field]
                elif field == "max_temp":
                    benchmark_results["max_heat"] = int(kwargs[field])
                elif field == "avg_temp":
                    benchmark_results["avg_heat"] = int(kwargs[field])
        
        # Add optional image benchmark fields
        optional_fields = [
            "gpu_power_watts", "gpu_memory_total", "platform", 
            "acceleration", "torch_version", "model_id"
        ]
        
    elif benchmark_type == "video_generation":
        # For video generation benchmarks (Wan2.1)
        required_fields = ["video_count", "max_temp", "avg_temp"]
        for field in required_fields:
            if field in kwargs:
                if field == "video_count":
                    benchmark_results["number_videos_generated"] = kwargs[field]
                elif field == "max_temp":
                    benchmark_results["max_heat"] = int(kwargs[field])
                elif field == "avg_temp":
                    benchmark_results["avg_heat"] = int(kwargs[field])
        
        # Add video-specific fields
        video_fields = ["total_frames", "frames_per_second", "model_size", "resolution"]
        for field in video_fields:
            if field in kwargs and kwargs[field] is not None:
                benchmark_results[field] = kwargs[field]
        
        # Add optional video benchmark fields
        optional_fields = [
            "gpu_power_watts", "gpu_memory_total", "platform", 
            "acceleration", "torch_version"
        ]
    
    # Add additional fields if provided
    for field in optional_fields:
        if field in kwargs and kwargs[field] is not None:
            benchmark_results[field] = kwargs[field]
    
    # Upload to Supabase using REST API
    try:
        # Direct REST API endpoint for the table
        api_url = f"{SUPABASE_URL}/rest/v1/benchmark"
        
        # Make the request with auth headers - change Prefer header to get response data
        response = requests.post(
            api_url,
            json=benchmark_results,
            headers={
                "Content-Type": "application/json",
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Prefer": "return=representation"  # Changed from minimal to get data back
            }
        )
        
        # Check if successful
        if response.status_code in (200, 201):
            # Parse the response to get the ID
            try:
                record_data = response.json()
                if isinstance(record_data, list) and len(record_data) > 0:
                    record_id = record_data[0].get('id')
                    print(f"✅ Results uploaded successfully to benchmark results!")
                    print(f"Your ID at www.unitedcompute.ai/gpu-benchmark: {record_id}")
                    return True, "Upload successful", record_id
                else:
                    return True, "Upload successful, but couldn't retrieve ID", None
            except Exception as e:
                return True, f"Upload successful, but error parsing response: {e}", None
        else:
            error_message = f"Error: {response.text}"
            print(f"❌ {error_message}")
            return False, error_message, None
            
    except Exception as e:
        error_message = f"Error uploading submitting to benchmark results: {e}"
        print(f"❌ {error_message}")
        print("\nTroubleshooting tips:")
        print("1. Check your network connection")
        return False, error_message, None