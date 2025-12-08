"""
Real-time Semantic Segmentation Inference Script
Uses pretrained models for scene labeling in autonomous vehicle scenarios.
"""

import os
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from helper import get_device
from bisenetv2_model import BiSeNetV2

# Load configuration
def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load pretrained model
def load_pretrained_model(config):
    """Load a pretrained segmentation model."""
    model_config = config['model']
    num_classes = config['dataset']['num_classes']
    architecture = model_config.get('architecture', 'bisenetv2').lower()
    
    device = get_device()
    
    # Choose architecture
    if architecture == 'bisenetv2':
        # Use BiSeNetV2 model
        model = BiSeNetV2(n_classes=num_classes, aux_mode='eval')
        
        # Load pretrained weights if available
        if os.path.exists(model_config['pretrained_path']):
            print(f"Loading pretrained BiSeNetV2 model from {model_config['pretrained_path']}")
            state_dict = torch.load(model_config['pretrained_path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Pretrained model not found at {model_config['pretrained_path']}")
            print("Using randomly initialized BiSeNetV2")
    else:
        # Fall back to segmentation_models_pytorch
        print(f"Using {architecture} from segmentation_models_pytorch")
        model = smp.FPN(
            encoder_name=model_config.get('encoder', 'mobilenet_v2'),
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        
        # Load pretrained weights if available
        if os.path.exists(model_config['pretrained_path']):
            print(f"Loading pretrained model from {model_config['pretrained_path']}")
            state_dict = torch.load(model_config['pretrained_path'], map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            print("Using ImageNet pretrained encoder only")
            model = smp.FPN(
                encoder_name=model_config.get('encoder', 'mobilenet_v2'),
                encoder_weights='imagenet',
                in_channels=3,
                classes=num_classes,
                activation=None,
            )
    
    model.to(device)
    model.eval()
    
    return model, device

# Preprocessing
def get_preprocessing(image_size):
    """Get preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Visualization
def visualize_segmentation(image, prediction, config, save_path=None):
    """Visualize segmentation results with color mapping."""
    color_palette = np.array(config['color_palette'], dtype=np.uint8)
    class_names = config['class_names']
    
    # Create colored segmentation mask using traditional method
    h, w = prediction.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(color_palette)):
        colored_mask[prediction == class_id] = color_palette[class_id]
    
    # Create masks for draw_segmentation_masks
    # Resize image to match prediction size
    image_resized = image.resize((w, h))
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1)
    
    # Create boolean masks for each class
    num_classes = len(color_palette)
    masks = torch.zeros((num_classes, h, w), dtype=torch.bool)
    for class_id in range(num_classes):
        masks[class_id] = torch.from_numpy(prediction == class_id)
    
    # Convert color palette to list of tuples for draw_segmentation_masks
    colors = [tuple(color) for color in color_palette]
    
    # Draw segmentation masks using torchvision utility
    overlay_tensor = draw_segmentation_masks(
        image_tensor,
        masks,
        alpha=0.4,
        colors=colors
    )
    overlay_torch = overlay_tensor.permute(1, 2, 0).numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay using draw_segmentation_masks
    axes[2].imshow(overlay_torch)
    axes[2].set_title('Overlay (torchvision)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return colored_mask

# Inference function
def predict_single_image(model, image_path, config, device):
    """Run inference on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    image_size = tuple(config['dataset']['image_size'])
    preprocess = get_preprocessing(image_size)
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        # Handle different output formats
        if isinstance(output, tuple):
            output = output[0]  # BiSeNetV2 in eval mode returns tuple
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Resize prediction to original image size
    prediction_pil = Image.fromarray(prediction.astype(np.uint8))
    prediction_resized = prediction_pil.resize(original_size, Image.NEAREST)
    prediction = np.array(prediction_resized)
    
    return image, prediction

# Calculate metrics
def calculate_class_distribution(prediction, config):
    """Calculate the distribution of predicted classes."""
    class_names = config['class_names']
    unique, counts = np.unique(prediction, return_counts=True)
    
    total_pixels = prediction.size
    print("\n=== Class Distribution ===")
    for class_id, count in zip(unique, counts):
        if class_id < len(class_names):
            percentage = (count / total_pixels) * 100
            print(f"{class_names[class_id]:15s}: {percentage:5.2f}%")

# Main inference function
def run_inference(image_path, config_path='configs/config.yaml', save_output=True):
    """Run inference on an image and visualize results."""
    # Load configuration
    config = load_config(config_path)
    
    # Load model
    model, device = load_pretrained_model(config)
    
    # Run inference
    print(f"\nProcessing: {image_path}")
    image, prediction = predict_single_image(model, image_path, config, device)
    
    # Calculate class distribution
    calculate_class_distribution(prediction, config)
    
    # Visualize results
    if config['inference']['visualization']:
        output_dir = config['inference']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        if save_output:
            image_name = os.path.basename(image_path).replace('.png', '_result.png')
            save_path = os.path.join(output_dir, image_name)
        else:
            save_path = None
        
        colored_mask = visualize_segmentation(image, prediction, config, save_path)
    
    return image, prediction

# Batch inference
def run_batch_inference(config_path='configs/config.yaml', split='val', num_samples=5):
    """Run inference on multiple images from the dataset."""
    config = load_config(config_path)
    
    # Load file list
    file_list_path = config['dataset'][f'{split}_list']
    with open(file_list_path, 'r') as f:
        lines = f.readlines()[:num_samples]
    
    # Load model
    model, device = load_pretrained_model(config)
    
    # Process each image
    for line in lines:
        image_path = line.strip().split()[0]
        print(f"\n{'='*60}")
        run_inference(image_path, config_path, save_output=True)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Inference on specific image
        image_path = sys.argv[1]
        run_inference(image_path)
    else:
        # Batch inference on validation set
        print("Running batch inference on 5 validation images...")
        run_batch_inference(split='val', num_samples=5)
