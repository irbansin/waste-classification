import io
import base64
from PIL import Image
import torch
import torch.nn.functional as F
from .transforms import get_transform


def classify_patch(patch_img, model, classes, transform=None):
    if transform is None:
        transform = get_transform()
    
    # Ensure RGB
    if isinstance(patch_img, Image.Image):
        patch_img = patch_img.convert("RGB")
    else:
        # If somehow not a PIL Image, convert defensively
        patch_img = Image.fromarray(patch_img).convert("RGB")

    # Move tensor to the model's device
    device = next(model.parameters()).device
    patch_tensor = transform(patch_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # TTA: Predict on original and flipped image
        outputs = model(patch_tensor)
        outputs_flip = model(torch.flip(patch_tensor, [3]))
        # Average predictions
        avg_probs = (F.softmax(outputs, dim=1) + F.softmax(outputs_flip, dim=1)) / 2
        probs = avg_probs[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_prob = float(probs[pred_idx].item())
    pred_class = classes[pred_idx]
        
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
