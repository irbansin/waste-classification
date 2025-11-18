import io
import base64
from PIL import Image
import torch
import torch.nn.functional as F
from .transforms import get_transform


def classify_patch(patch_img, model, classes, transform=None):
    if transform is None:
        transform = get_transform()
    patch_tensor = transform(patch_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(patch_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = classes[pred_idx]
        pred_prob = probs[pred_idx].item()
    # Encode patch to base64
    patch_bytes = io.BytesIO()
    patch_img.save(patch_bytes, format="JPEG")
    patch_b64 = base64.b64encode(patch_bytes.getvalue()).decode()
    return {
        "predicted_label": pred_class,
        "probability": pred_prob,
        "patch_b64": patch_b64
    }


def classify_patch_hierarchical(patch_img, model, classes, transform=None, hierarchy=None):
    # The hierarchical API has been removed. Use `classify_patch` instead.
    raise NotImplementedError("classify_patch_hierarchical has been removed. Use classify_patch(patch_img, model, classes, transform) instead.")
