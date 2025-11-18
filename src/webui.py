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
from src.utils.classification import classify_patch
import torch.nn as nn
from torchvision import models as tv_models
from typing import Optional

# --- Small helpers to keep the main code clean ---
def _resolve_checkpoint(config) -> Optional[str]:
    """Return the first existing checkpoint path among config and defaults."""
    # 1. Check user selection
    if 'selected_model_path' in st.session_state and st.session_state['selected_model_path']:
        p = st.session_state['selected_model_path']
        if os.path.exists(p):
            return p

    # 2. Check config and defaults
    candidates = [
        config.get('checkpoint_path'),
        'outputs/trash_classifier.pth',
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _inspect_checkpoint(ckpt_path):
    """Load checkpoint and return (state_dict, meta_dict)."""
    try:
        state = torch.load(ckpt_path, map_location='cpu')
        # Check if it's a metadata dict
        meta = {}
        if isinstance(state, dict) and 'state_dict' in state:
            meta = state
            state_dict = state['state_dict']
        else:
            state_dict = state
        return state_dict, meta
    except Exception:
        return None, None


def _infer_arch_and_classes(state_dict):
    """Heuristically determine architecture and num_classes from state_dict keys."""
    arch = 'resnet18'  # fallback
    num_classes = 10   # fallback

    keys = list(state_dict.keys())

    # Detect Arch
    if any(k.startswith('features.') for k in keys):
        arch = 'mobilenet_v2'
        # Find classifier weight
        for k in keys:
            if 'classifier.1.weight' in k:
                num_classes = state_dict[k].shape[0]
                break
    elif any(k.startswith('layer') for k in keys):
        # ResNet
        if 'layer1.0.conv3.weight' in keys:
            arch = 'resnet50'
        else:
            arch = 'resnet18'

        if 'fc.weight' in keys:
            num_classes = state_dict['fc.weight'].shape[0]

    return arch, num_classes


def _load_classifier(num_classes: int, ckpt_path: Optional[str], arch: str = 'auto'):
    """Create a model and load checkpoint. Auto-detects arch if 'auto'."""
    state_dict = None
    meta = {}

    # 1. Load state_dict first if we have a path
    if ckpt_path and os.path.exists(ckpt_path) and not ckpt_path.endswith('.pt'):
        state_dict, meta = _inspect_checkpoint(ckpt_path)

    # 2. Determine Architecture and Num Classes
    if arch == 'auto':
        if meta and 'arch' in meta:
            arch = meta['arch']
        elif state_dict:
            arch, inferred_classes = _infer_arch_and_classes(state_dict)
            # Prefer inferred classes if we are auto-detecting
            num_classes = inferred_classes
        else:
            arch = 'resnet18' # default fallback

    # 3. Update session state classes if metadata provided them
    if meta and 'classes' in meta:
        st.session_state['rt_classes'] = meta['classes']
        num_classes = len(meta['classes'])

    # 4. Instantiate Model
    if arch == 'resnet50':
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'mobilenet_v2':
        model = tv_models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        # Default to resnet18
        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 5. Load Weights
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=False)
            st.toast(f"Loaded {arch} with {num_classes} classes.")
        except Exception as e:
            st.warning(f"Could not load weights into {arch}: {e}")
    elif ckpt_path:
        st.warning("Checkpoint path provided but could not load state_dict.")
    else:
        st.warning("No checkpoint found. Using randomly initialized weights.")

    return model.eval()


def _maybe_reload_model(config) -> None:
    """Reloads the classifier in session_state if the checkpoint file changed.

    This checks the path returned by _resolve_checkpoint(config). If the file's
    modification time differs from the last seen mtimes stored in
    `st.session_state['rt_model_mtime']`, the function attempts to load a new
    model and replace `st.session_state['rt_model']` so the running app picks up
    the new weights immediately.
    """
    try:
        ckpt = _resolve_checkpoint(config)
        if not ckpt:
            return
        mtime = os.path.getmtime(ckpt)
        prev = st.session_state.get('rt_model_mtime')
        # If changed (or first time record), reload
        if prev is None or mtime != prev:
            st.info(f"Detected updated model checkpoint: {os.path.basename(ckpt)}. Reloading...")
            arch = st.session_state.get('selected_arch', 'auto')
            new_model = _load_classifier(len(st.session_state['rt_classes']), ckpt, arch=arch)
            if new_model is not None:
                st.session_state['rt_model'] = new_model
                st.session_state['rt_model_mtime'] = mtime
                st.success("Model reloaded successfully.")
    except Exception as e:
        # Do not raise — keep running with the previous model and show a warning
        st.warning(f"Error while attempting to hot-reload model: {e}")


def _list_models(directory="outputs") -> list:
    """List all .pth, .pt, .h5 files in the directory."""
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.endswith(('.pth', '.pt', '.h5'))])


def _init_runtime():
    """Initialize config, classifier, YOLO, and transform in session_state once."""
    if 'rt_config' not in st.session_state:
        with open("configs/baseline.yaml", 'r') as f:
            st.session_state['rt_config'] = yaml.safe_load(f)
    config = st.session_state['rt_config']
    st.session_state['rt_classes'] = config['classes']
    # No hierarchical classification anymore
    # Classifier
    if 'rt_model' not in st.session_state or st.session_state['rt_model'] is None:
        try:
            ckpt_path = _resolve_checkpoint(config)
            arch = st.session_state.get('selected_arch', 'auto')
            st.session_state['rt_model'] = _load_classifier(len(st.session_state['rt_classes']), ckpt_path, arch=arch)
        except Exception as e:
            st.error(f"Failed to load classification model: {e}")
            st.session_state['rt_model'] = None
    # YOLO
    if 'rt_yolo' not in st.session_state or st.session_state['rt_yolo'] is None:
        try:
            st.session_state['rt_yolo'] = YOLO('yolov8n.pt')
        except Exception as e:
            st.error(f"Failed to load YOLO model: {e}")
            st.session_state['rt_yolo'] = None
    # Transform
    if 'rt_transform' not in st.session_state:
        st.session_state['rt_transform'] = get_transform()
    return True


# Model Selection Dropdown
col1, col2 = st.columns(2)

available_models = _list_models()
with col1:
    if available_models:
        # Default to the first one or keep previous selection if valid
        idx = 0
        current_path = st.session_state.get('selected_model_path')
        if current_path:
            current_name = os.path.basename(current_path)
            if current_name in available_models:
                idx = available_models.index(current_name)

        selected_model = st.selectbox("Select Classification Model", available_models, index=idx)
        st.session_state['selected_model_path'] = os.path.join("outputs", selected_model)
    else:
        st.warning("No models found in 'outputs/' directory.")

with col2:
    arch_options = ['auto', 'resnet18', 'resnet50', 'mobilenet_v2']
    selected_arch = st.selectbox("Select Model Architecture", arch_options, index=0)
    # If architecture changes, force reload
    if 'selected_arch' in st.session_state and st.session_state['selected_arch'] != selected_arch:
        st.session_state['rt_model'] = None # Force reload
    st.session_state['selected_arch'] = selected_arch


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
    if yolo_model is None or model is None:
        camera_placeholder.error("Models failed to initialize.")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            camera_placeholder.error("Could not open camera.")
        else:
            while st.session_state['camera_running']:
                # Check for model updates (hot-swap)
                _maybe_reload_model(st.session_state['rt_config'])

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
                        # Flat classification (no hierarchy)
                        result = classify_patch(crop, model, classes, transform)
                        label = result['predicted_label']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"{label} ({result['probability']:.2f})"
                        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                camera_placeholder.image(frame, channels="BGR", caption="Live Trash Classification", width='stretch')
                time.sleep(0.03)  # ~30 FPS
            cap.release()
            camera_placeholder.info("Camera stopped.")
