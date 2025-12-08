"""
ONNX Runtime inference engine for optimized real-time segmentation.
"""

import numpy as np
import cv2
import onnxruntime as ort


class ONNXSegmentation:
    def __init__(
        self,
        onnx_path='pretrained_models/bisenetv2.onnx',
        input_size=(512, 1024),
        num_classes=19
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        
        # Try to use CoreML on macOS
        try:
            if 'CoreMLExecutionProvider' in ort.get_available_providers():
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        except:
            pass
        
        print(f"Available providers: {ort.get_available_providers()}")
        print(f"Using providers: {providers}")
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Input: {self.input_name}")
        print(f"Output: {self.output_name}")
    
    def preprocess(self, image):
        """Preprocess image for ONNX inference."""
        # Resize
        img = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
    
    def predict(self, image):
        """Run inference on image."""
        h, w = image.shape[:2]
        
        # Preprocess
        input_data = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        output = outputs[0]
        
        # Handle different output formats
        if isinstance(output, tuple):
            output = output[0]
        
        # Get prediction
        if len(output.shape) == 4:
            prediction = np.argmax(output[0], axis=0)
        else:
            prediction = np.argmax(output, axis=1)[0]
        
        # Resize to original size
        prediction = cv2.resize(prediction.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        return prediction


def benchmark_onnx(num_frames=100):
    """Benchmark ONNX inference performance."""
    import time
    
    segmenter = ONNXSegmentation()
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    print(f"\nRunning ONNX benchmark on {num_frames} frames...")
    print(f"Frame size: 720x1280")
    
    # Warmup
    for _ in range(10):
        _ = segmenter.predict(dummy_frame)
    
    # Benchmark
    start_time = time.time()
    
    for i in range(num_frames):
        mask = segmenter.predict(dummy_frame)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            print(f"Frame {i+1}/{num_frames}: {fps:.2f} FPS")
    
    total_time = time.time() - start_time
    avg_fps = num_frames / total_time
    
    print(f"\nONNX Benchmark Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Latency: {1000/avg_fps:.2f}ms per frame")


if __name__ == '__main__':
    benchmark_onnx()
