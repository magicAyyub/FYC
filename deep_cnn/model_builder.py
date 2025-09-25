import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class MyCNN(nn.Module):
    """Builds a CNN using a pre-trained neural network replacing
    the final layers for a custom image task

    Args:
        model_base: Model choice for pre-trained network
        input_shape: 3D shape of input tensor
    """

    def __init__(
        self, model_base="resnet101", input_shape=(3, 224, 224), n_classes=365
    ):
        super().__init__()
        self.pretrained = initialise_model(model_base)  # load pre-trained model
        dim = get_final_dimension(self.pretrained, input_shape)
        self.my_new_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(dim, n_classes)  # add fully connected layer
        )

    def forward(self, x):
        x = x.float()
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        return x

    def count_params(self):
        pytorch_total_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return pytorch_total_params


def initialise_model(model_base):
    """Function to initiliase pre-trained model and remove final layers"""
    method_to_call = getattr(models, model_base)
    
    # Use new weights API instead of deprecated pretrained parameter
    if model_base == "resnet101":
        try:
            from torchvision.models import ResNet101_Weights
            model = method_to_call(weights=ResNet101_Weights.IMAGENET1K_V1)
        except ImportError:
            # Fallback for older torchvision versions
            model = method_to_call(pretrained=True)
    elif model_base == "resnet18":
        try:
            from torchvision.models import ResNet18_Weights
            model = method_to_call(weights=ResNet18_Weights.IMAGENET1K_V1)
        except ImportError:
            # Fallback for older torchvision versions
            model = method_to_call(pretrained=True)
    else:
        # For other models, use DEFAULT weights if available
        try:
            model = method_to_call(weights="DEFAULT")
        except TypeError:
            # Fallback for older torchvision versions
            model = method_to_call(pretrained=True)
    
    model = nn.Sequential(*(list(model.children())[:-1]))  # remove final layers
    return model


def get_final_dimension(model, input_shape):
    """Calculates output of pre-trained model given image input_shape"""
    x = torch.randn((1,) + input_shape)
    out = model(x)
    return np.prod(out.shape)
