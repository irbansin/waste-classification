"""
Streamlit Web UI for Waste Image Classification
Allows users to upload an image, select a model, and view predictions interactively.
"""
import streamlit as st
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


st.set_page_config(page_title="Waste Image Classifier", layout="centered")
st.title("Waste Image Classification Web UI")




st.header("Live Camera Trash Classification")
st.markdown("""
This mode processes your camera feed in real time, detects and classifies trash,
and overlays the class and hierarchy on the video. Use this as the decision engine for a robot.
""")


import cv2
import torch
import yaml
import time
from ultralytics import YOLO
from src.utils.transforms import get_transform
from src.utils.classification import classify_patch_hierarchical
import torch.nn as nn
from torchvision import models as tv_models
from typing import Optional

# --- Small helpers to keep the main code clean ---
def _resolve_checkpoint(config) -> Optional[str]:
    """Return the first existing checkpoint path among config and defaults."""
    candidates = [
        config.get('checkpoint_path'),
        'outputs/baseline_cnn_best.pt',
        'outputs/baseline_cnn_best.pth',
        'outputs/baseline_cnn_best.pth.tar',
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _load_classifier(num_classes: int, ckpt_path: Optional[str]):
    """Load a classifier. Prefer TorchScript; else use ResNet18 with optional state_dict."""
    # Try TorchScript
    if ckpt_path and ckpt_path.endswith('.pt'):
        try:
            model = torch.jit.load(ckpt_path, map_location='cpu').eval()
            return model
        except Exception:
            pass
    # Fallback to torchvision ResNet18 head
    base = tv_models.resnet18(weights=None)
    base.fc = nn.Linear(base.fc.in_features, num_classes)
    if ckpt_path and not ckpt_path.endswith('.pt'):
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            base.load_state_dict(state, strict=False)
        except Exception:
            st.warning("Could not load checkpoint into ResNet18. Using randomly initialized weights.")
    elif not ckpt_path:
        st.warning("No checkpoint found. Using randomly initialized weights.")
    return base.eval()

def _init_runtime():
    """Initialize config, classifier, YOLO, and transform in session_state once."""
    if 'rt_config' not in st.session_state:
        with open("configs/baseline.yaml", 'r') as f:
            st.session_state['rt_config'] = yaml.safe_load(f)
    config = st.session_state['rt_config']
    st.session_state['rt_classes'] = config['classes']
    st.session_state['rt_hierarchy'] = config.get('hierarchy', {})
    # Classifier
    if 'rt_model' not in st.session_state:
        try:
            ckpt_path = _resolve_checkpoint(config)
            st.session_state['rt_model'] = _load_classifier(len(st.session_state['rt_classes']), ckpt_path)
        except Exception as e:
            st.error(f"Failed to load classification model: {e}")
            st.session_state['rt_model'] = None
    # YOLO
    if 'rt_yolo' not in st.session_state:
        try:
            st.session_state['rt_yolo'] = YOLO('yolov8n.pt')
        except Exception as e:
            st.error(f"Failed to load YOLO model: {e}")
            st.session_state['rt_yolo'] = None
    # Transform
    if 'rt_transform' not in st.session_state:
        st.session_state['rt_transform'] = get_transform()
    return True

camera_placeholder = st.empty()
status_placeholder = st.empty()

if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = False

run_camera = st.button("Start Live Camera Feed")
stop_camera = st.button("Stop Camera Feed")

if run_camera:
    st.session_state['camera_running'] = True
    status_placeholder.success("Camera feed started.")
if stop_camera:
    st.session_state['camera_running'] = False
    status_placeholder.info("Camera feed stopped.")

if st.session_state['camera_running']:
    with st.spinner("Initializing models..."):
        _init_runtime()
    model = st.session_state['rt_model']
    yolo_model = st.session_state['rt_yolo']
    transform = st.session_state['rt_transform']
    classes = st.session_state['rt_classes']
    hierarchy = st.session_state['rt_hierarchy']
    if yolo_model is None or model is None:
        camera_placeholder.error("Models failed to initialize.")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            camera_placeholder.error("Could not open camera.")
        else:
            while st.session_state['camera_running']:
                ret, frame = cap.read()
                if not ret:
                    camera_placeholder.error("Failed to read frame from camera.")
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yolo_results = yolo_model(rgb_frame)
                if hasattr(yolo_results[0], 'boxes') and hasattr(yolo_results[0].boxes, 'xyxy'):
                    detections = yolo_results[0].boxes.xyxy.cpu().numpy()
                    for box in detections:
                        x1, y1, x2, y2 = map(int, box[:4])
                        crop = Image.fromarray(rgb_frame[y1:y2, x1:x2])
                        result = classify_patch_hierarchical(crop, model, classes, transform, hierarchy)
                        label = result['hierarchy']['level_2']
                        level_1 = result['hierarchy']['level_1']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"{level_1}/{label} ({result['probability']:.2f})"
                        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                camera_placeholder.image(frame, channels="BGR", caption="Live Trash Classification", use_column_width=True)
                time.sleep(0.03)  # ~30 FPS
            cap.release()
            camera_placeholder.info("Camera stopped.")
