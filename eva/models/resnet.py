import torch.nn as nn
import torchvision.models as models

def get_resnet_model(num_classes=6, pretrained=True, weights_path=None):
    """
    Returns a ResNet-50 model for image classification.
    - num_classes: number of output classes (sets the final fully connected layer)
    - pretrained: if True, loads ImageNet weights
    - weights_path: if given, loads additional weights from this path
    Handles errors gracefully and returns the model ready for training or inference.
    """
    try:
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    except Exception as e:
        raise RuntimeError(f"Could not load ResNet model: {e}")
    return model
