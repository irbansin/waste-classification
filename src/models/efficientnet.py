import timm
import torch.nn as nn

def get_efficientnet_model(num_classes=6, model_name='efficientnet_b0', pretrained=True, weights_path=None):
    """
    Returns an EfficientNet model for image classification.
    - num_classes: number of output classes (sets the final classification layer)
    - model_name: EfficientNet variant (e.g., 'efficientnet_b0')
    - pretrained: if True, loads ImageNet weights
    - weights_path: if given, loads additional weights from this path
    Handles errors gracefully and returns the model ready for training or inference.
    """
    try:
        model = timm.create_model(model_name, pretrained=pretrained)
        in_features = model.classifier.in_features if hasattr(model, 'classifier') else model.head.in_features
        if hasattr(model, 'classifier'):
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            model.head = nn.Linear(in_features, num_classes)
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    except Exception as e:
        raise RuntimeError(f"Could not load EfficientNet model: {e}")
    return model
