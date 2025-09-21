"""
Streamlit Web UI for Waste Image Classification
Allows users to upload an image, select a model, and view predictions interactively.
"""
import streamlit as st
import requests
from PIL import Image
import io
import base64

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

API_URL = "http://localhost:8001"

st.set_page_config(page_title="Waste Image Classifier", layout="centered")
st.title("Waste Image Classification Web UI")

# Tabs for single and multi-item detection/classification
# mode = st.sidebar.radio("Choose Mode", [
#     "Classify Single Waste Item",
#     "Detect & Classify Multiple Items",
#     "Live Camera Trash Classification"
# ])


st.header("Live Camera Trash Classification")
st.markdown("""
This mode processes your camera feed in real time, detects and classifies trash,
and overlays the class and hierarchy on the video. Use this as the decision engine for a robot.
""")
# st.warning("Live video feed will open in a separate window. Close it or press 'q' to stop.")
    # if st.button("Start Live Camera Processing"):
    #     import subprocess
    #     st.info("Launching real-time decision engine...")
    #     # Launch the external script as a subprocess
    #     proc = subprocess.Popen(["python", "realtime_decision_engine.py"])
    #     st.success("Real-time camera processing started. Check the new window.")
    #     st.info("To stop, close the camera window or press 'q'.")

import cv2
import numpy as np
import torch
from PIL import Image
import yaml
from ultralytics import YOLO
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import get_resnet_model
from src.models.efficientnet import get_efficientnet_model
from src.utils.transforms import get_transform
from src.utils.classification import classify_patch_hierarchical
import time

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
    # Load config and models only once
    if 'rt_config' not in st.session_state:
        with open("configs/baseline.yaml", 'r') as f:
            st.session_state['rt_config'] = yaml.safe_load(f)
    config = st.session_state['rt_config']
    classes = config['classes']
    hierarchy = config.get('hierarchy', {})
    if 'rt_model' not in st.session_state:
        model_type = config.get('model', 'baseline_cnn')
        if model_type == 'baseline_cnn':
            model = BaselineCNN(num_classes=len(classes))
        elif model_type == 'resnet50':
            model = get_resnet_model(num_classes=len(classes))
        elif model_type == 'efficientnet_b0':
            model = get_efficientnet_model(num_classes=len(classes), model_name='efficientnet_b0')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        # Resolve model weights path robustly
        import glob
        checkpoint_dir = config.get('checkpoint_dir', 'outputs')
        configured_model_path = config.get('model_path')
        env_model_path = os.getenv('MODEL_PATH')
        model_path = env_model_path or configured_model_path or os.path.join(checkpoint_dir, 'baseline_cnn_best.pth')
        if not os.path.exists(model_path):
            # Fallback: pick the most recent .pth in checkpoint_dir
            candidates = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
            if candidates:
                model_path = max(candidates, key=os.path.getmtime)
        if not os.path.exists(model_path):
            st.error(f"Model weights not found. Expected at '{model_path}'. Set env var MODEL_PATH or add 'model_path' in configs/baseline.yaml, or place a .pth file under '{checkpoint_dir}/'.")
            st.stop()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        st.session_state['rt_model'] = model
    model = st.session_state['rt_model']
    if 'rt_yolo' not in st.session_state:
        st.session_state['rt_yolo'] = YOLO('yolov8n.pt')
    yolo_model = st.session_state['rt_yolo']
    if 'rt_transform' not in st.session_state:
        st.session_state['rt_transform'] = get_transform()
    transform = st.session_state['rt_transform']
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
                    conf = box[4] if len(box) > 4 else 1.0
                    crop = Image.fromarray(rgb_frame[y1:y2, x1:x2])
                    result = classify_patch_hierarchical(crop, model, classes, transform, hierarchy)
                    label = result['hierarchy']['level_2']
                    level_1 = result['hierarchy']['level_1']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{level_1}/{label} ({result['probability']:.2f})"
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            camera_placeholder.image(frame, channels="BGR", caption="Live Trash Classification", width='stretch')
            time.sleep(0.03)  # ~30 FPS
        cap.release()
        camera_placeholder.info("Camera stopped.")
