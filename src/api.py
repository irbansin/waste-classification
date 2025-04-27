"""
FastAPI-based web API for Waste Image Classification models.
Allows uploading an image and returns predicted class label (and probability).
Supports loading any of the trained models (baseline CNN, ResNet, EfficientNet).
"""
import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import get_resnet_model
from src.models.efficientnet import get_efficientnet_model
from ultralytics import YOLO
import yaml
import base64

app = FastAPI(title="Waste Image Classification API")

# --- CONFIG ---
MODEL_TYPE = "resnet50"  # Change to 'baseline_cnn' or 'efficientnet_b0' as needed
MODEL_PATH = "outputs/resnet50_best.pth"  # Path to trained weights
CONFIG_PATH = "configs/baseline.yaml"  # Path to config (for class names)
IMG_SIZE = 224

# --- Load config and model ---
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
classes = config['classes']
num_classes = len(classes)

def load_model():
    if MODEL_TYPE == 'baseline_cnn':
        model = BaselineCNN(num_classes=num_classes)
    elif MODEL_TYPE == 'resnet50':
        model = get_resnet_model(num_classes=num_classes)
    elif MODEL_TYPE == 'efficientnet_b0':
        model = get_efficientnet_model(num_classes=num_classes, model_name='efficientnet_b0')
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# --- Load YOLOv8 object detector (COCO-pretrained) ---
yolo_model = YOLO('yolov8n.pt')  # Use nano for speed, can switch to yolov8s.pt for better accuracy

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, returns predicted class and probability.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")
    img_tensor = transform(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_label = classes[pred_idx]
    return JSONResponse({
        "predicted_label": pred_label,
        "probability": float(probs[pred_idx]),
        "class_probs": {c: float(p) for c, p in zip(classes, probs)}
    })

@app.post("/detect_and_classify")
async def detect_and_classify(file: UploadFile = File(...)):
    """
    Detects multiple objects in an image (using YOLOv8), crops each, classifies with waste classifier,
    and returns bounding boxes, predicted classes, probabilities, and base64-encoded cropped images.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")
    # YOLO expects numpy array
    import numpy as np
    np_img = np.array(image)
    results = yolo_model(np_img)
    detections = results[0].boxes.xyxy.cpu().numpy()  # xyxy format
    scores = results[0].boxes.conf.cpu().numpy()
    classes_yolo = results[0].boxes.cls.cpu().numpy().astype(int)
    # Filter for "waste-like" classes (optional: keep all for now)
    crops = []
    outputs = []
    for i, (xyxy, score, yolo_cls) in enumerate(zip(detections, scores, classes_yolo)):
        x1, y1, x2, y2 = map(int, xyxy)
        if score < 0.3:
            continue  # Skip low confidence
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)
        # Classify the cropped image
        crop_tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            logits = model(crop_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            pred_label = classes[pred_idx]
        # Encode crop to base64
        buffered = io.BytesIO()
        crop.save(buffered, format="JPEG")
        crop_b64 = base64.b64encode(buffered.getvalue()).decode()
        outputs.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "score": float(score),
            "yolo_class": int(yolo_cls),
            "predicted_label": pred_label,
            "probability": float(probs[pred_idx]),
            "class_probs": {c: float(p) for c, p in zip(classes, probs)},
            "crop_b64": crop_b64
        })
    # Optionally, return the original image with boxes (as base64)
    import cv2
    img_with_boxes = np_img.copy()
    for out in outputs:
        x1, y1, x2, y2 = out["box"]
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0,255,0), 2)
    _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    img_with_boxes_b64 = base64.b64encode(img_encoded.tobytes()).decode()
    return JSONResponse({
        "detections": outputs,
        "image_with_boxes_b64": img_with_boxes_b64
    })

@app.get("/")
def root():
    return {"message": "Waste Image Classification API. Use POST /predict with an image file."}
