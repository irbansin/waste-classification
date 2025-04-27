import io
import base64
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# This should be set to match your model's expected input
IMG_SIZE = 224

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
    """
    Hierarchical classification: first Organic/Inorganic, then subclass if Inorganic.
    Uses hierarchy mapping from config.
    """
    if transform is None:
        transform = get_transform()
    if hierarchy is None:
        hierarchy = {}
    patch_tensor = transform(patch_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(patch_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = classes[pred_idx]
        pred_prob = probs[pred_idx].item()
    # Dynamic hierarchy logic
    level_1 = hierarchy.get(pred_class, 'Unknown')
    hierarchy_dict = {
        'level_1': level_1,
        'level_2': pred_class
    }
    # Encode patch to base64
    patch_bytes = io.BytesIO()
    patch_img.save(patch_bytes, format="JPEG")
    patch_b64 = base64.b64encode(patch_bytes.getvalue()).decode()
    return {
        "predicted_label": pred_class,
        "probability": pred_prob,
        "patch_b64": patch_b64,
        "hierarchy": hierarchy_dict
    }

def divide_image_into_grid(pil_img, grid_size=3):
    w, h = pil_img.size
    patch_w, patch_h = w // grid_size, h // grid_size
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = i * patch_w
            upper = j * patch_h
            right = (i + 1) * patch_w if i < grid_size - 1 else w
            lower = (j + 1) * patch_h if j < grid_size - 1 else h
            patch = pil_img.crop((left, upper, right, lower))
            patches.append({
                "patch": patch,
                "coords": [int(left), int(upper), int(right), int(lower)]
            })
    return patches
