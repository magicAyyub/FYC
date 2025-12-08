# Real-Time Scene Labeling for Autonomous Vehicles

BiSeNetV2-based semantic segmentation for real-time scene understanding in autonomous driving scenarios, trained on the Cityscapes dataset.

## Project Overview

This project implements real-time semantic segmentation using BiSeNetV2, a state-of-the-art model designed specifically for autonomous driving applications. The model segments road scenes into 19 semantic classes including roads, vehicles, pedestrians, and infrastructure.

cityscapes : leftImg8bit & gtFine

### Key Features
- Real-time Performance: 12.8MB model optimized for speed
- 19 Semantic Classes: Comprehensive scene understanding
- Pretrained Model: Ready-to-use weights from Kaggle
- MPS/CUDA Support: GPU acceleration on macOS and Linux
- Easy Integration: Simple Python API for inference

## Quick Start

### Installation
```bash
# Install dependencies
uv sync
```


### Run Inference

**Real-time Webcam:**
```bash
# Use built-in webcam
uv run python realtime_inference.py --mode webcam

# Use iPhone as webcam (with Camo or Continuity Camera)
uv run python realtime_inference.py --mode webcam --camera 2
```

**Process Video:**
```bash
uv run python realtime_inference.py --mode video --input video.mp4 --output result.mp4
```

**Live Stream (RTSP/RTMP):**
```bash
uv run python realtime_inference.py --mode stream --input rtsp://localhost:8554/mystream
```

**Single Image:**
```bash
uv run python inference.py <path_to_image>
```

**Batch Processing:**
```bash
uv run python inference.py
```


## Getting Pretrained Model

1. Visit: https://www.kaggle.com/code/agampy/adversarial-patch-baseline/notebook
2. Download the BiSeNetV2 pretrained model
3. Place in: `pretrained_models/bisenetv2_cityscapes.pth`


## References

- Kaggle Notebook: [Adversarial Patch Baseline](https://www.kaggle.com/code/agampy/adversarial-patch-baseline/notebook)
- BiSeNetV2 Paper: "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
- Cityscapes Dataset: https://www.cityscapes-dataset.com/

## Contributing

This project is based on research in adversarial robustness and real-time segmentation for autonomous vehicles.

## License

See the original Kaggle notebook for licensing information.
