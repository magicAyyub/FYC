"""
Benchmark real-time inference performance.
"""

import time
import torch
import numpy as np
import cv2
from realtime_inference import RealtimeSegmentation


def benchmark_fps(config_path='configs/config.yaml', num_frames=100):
    """Measure inference FPS on dummy frames."""
    
    segmenter = RealtimeSegmentation(config_path)
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    print(f"Running benchmark on {num_frames} frames...")
    print(f"Frame size: 720x1280")
    print(f"Device: {segmenter.device}")
    
    # Warmup
    for _ in range(10):
        _ = segmenter.predict_frame(dummy_frame)
    
    # Benchmark
    start_time = time.time()
    
    for i in range(num_frames):
        mask = segmenter.predict_frame(dummy_frame)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            print(f"Frame {i+1}/{num_frames}: {fps:.2f} FPS")
    
    total_time = time.time() - start_time
    avg_fps = num_frames / total_time
    
    print(f"\nBenchmark Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Latency: {1000/avg_fps:.2f}ms per frame")


if __name__ == '__main__':
    benchmark_fps()
