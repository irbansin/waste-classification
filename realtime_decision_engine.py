import cv2
import numpy as np
import torch
import os
import glob
from PIL import Image
from ultralytics import YOLO
import yaml
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import get_resnet_model
from src.models.efficientnet import get_efficientnet_model
from src.utils.transforms import get_transform
from src.utils.classification import classify_patch_hierarchical

# --- Load config and model ---
CONFIG_PATH = "configs/baseline.yaml"
MODEL_TYPE = "baseline_cnn"  # Change as needed
MODEL_PATH = "outputs/baseline_cnn_best.pth"
IMG_SIZE = 224

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_type, num_classes):
    # Load config locally for path resolution
    config = load_config()
    if model_type == 'baseline_cnn':
        model = BaselineCNN(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = get_resnet_model(num_classes=num_classes)
    elif model_type == 'efficientnet_b0':
        model = get_efficientnet_model(num_classes=num_classes, model_name='efficientnet_b0')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # Resolve model weights path robustly
    checkpoint_dir = config.get('checkpoint_dir', 'outputs')
    configured_model_path = config.get('model_path')
    env_model_path = os.getenv('MODEL_PATH')
    model_path = env_model_path or configured_model_path or MODEL_PATH
    if not os.path.exists(model_path):
        # Fallback to newest .pth in checkpoint_dir
        candidates = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
        if candidates:
            model_path = max(candidates, key=os.path.getmtime)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found. Expected at '{model_path}'. Set env var MODEL_PATH or add 'model_path' in {CONFIG_PATH}, or place a .pth file under '{checkpoint_dir}/'.")
        raise SystemExit(1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def main():
    config = load_config()
    classes = config['classes']
    hierarchy = config.get('hierarchy', {})
    model = load_model(MODEL_TYPE, num_classes=len(classes))
    yolo_model = YOLO('yolov8n.pt')
    transform = get_transform()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect trash in the frame using YOLO
        yolo_results = yolo_model(rgb_frame)
        if hasattr(yolo_results[0], 'boxes') and hasattr(yolo_results[0].boxes, 'xyxy'):
            detections = yolo_results[0].boxes.xyxy.cpu().numpy()
            for box in detections:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4] if len(box) > 4 else 1.0
                # Crop detected region
                crop = Image.fromarray(rgb_frame[y1:y2, x1:x2])
                # Hierarchical classification
                result = classify_patch_hierarchical(crop, model, classes, transform, hierarchy)
                label = result['hierarchy']['level_2']
                level_1 = result['hierarchy']['level_1']
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{level_1}/{label} ({result['probability']:.2f})"
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Trash Classification (Press q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
