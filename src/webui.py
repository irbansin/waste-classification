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
Use the sliders in the sidebar to tune detection and classification thresholds if objects are missed.
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
    """Return the path to the required checkpoint if it exists."""
    p = 'outputs/trash_classifier_meta.pth'
    if os.path.exists(p):
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


def _load_classifier(num_classes: int, ckpt_path: Optional[str]):
    """Create a resnet50 model and load checkpoint."""
    state_dict = None
    meta = {}
    arch = 'resnet50'

    # 1. Load state_dict first if we have a path
    if ckpt_path and os.path.exists(ckpt_path) and not ckpt_path.endswith('.pt'):
        state_dict, meta = _inspect_checkpoint(ckpt_path)

    # 2. Update session state classes if metadata provided them
    if meta and 'classes' in meta:
        st.session_state['rt_classes'] = meta['classes']
        num_classes = len(meta['classes'])

    # 3. Instantiate Model
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # 4. Load Weights
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
            new_model = _load_classifier(len(st.session_state['rt_classes']), ckpt)
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
            st.session_state['rt_model'] = _load_classifier(len(st.session_state['rt_classes']), ckpt_path)
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


# Model Selection Info
st.info("This app is configured to use `outputs/trash_classifier_meta.pth`.")
# The selected_model_path is now implicitly handled by _resolve_checkpoint
# which is hardcoded to look for the specific meta file.
st.session_state['selected_model_path'] = 'outputs/trash_classifier_meta.pth'




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
    # Sidebar controls for runtime tuning
    yolo_conf = st.sidebar.slider("YOLO detection confidence", 0.05, 0.9, 0.25, 0.05,
                                  help="Lower this to detect more objects (but may add noise). Raise to be stricter.")
    cls_conf = st.sidebar.slider("Classification confidence threshold", 0.50, 0.95, 0.70, 0.01,
                                 help="Only show classification labels above this probability.")
    show_stats = st.sidebar.checkbox("Show detection stats", value=True)
    fallback_on_empty = st.sidebar.checkbox("Fallback classify full frame if no detections", value=True)
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
                # Run YOLO with adjustable confidence
                try:
                    yolo_results = yolo_model(rgb_frame, conf=yolo_conf)
                except TypeError:
                    # Older ultralytics versions may not accept conf in call; fallback
                    yolo_results = yolo_model(rgb_frame)
                if hasattr(yolo_results[0], 'boxes') and hasattr(yolo_results[0].boxes, 'xyxy'):
                    boxes = yolo_results[0].boxes
                    detections = boxes.xyxy.cpu().numpy()
                    # YOLO confidence threshold to reduce false detections
                    try:
                        confs = boxes.conf.detach().cpu().numpy()
                    except Exception:
                        confs = [1.0] * len(detections)

                    h, w = rgb_frame.shape[:2]
                    kept = 0
                    confidences = []
                    for box, det_conf in zip(detections, confs):
                        # Filter low-confidence detections from YOLO
                        if det_conf < yolo_conf:
                            continue
                        x1, y1, x2, y2 = map(int, box[:4])
                        # Clamp to frame bounds
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(0, min(x2, w))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(0, min(y2, h))
                        if x2 - x1 < 4 or y2 - y1 < 4:
                            continue

                        crop = Image.fromarray(rgb_frame[y1:y2, x1:x2])
                        # Flat classification (no hierarchy)
                        try:
                            result = classify_patch(crop, model, classes, transform)
                            conf = float(result.get('probability', 0.0))
                            if conf >= cls_conf:
                                label = result['predicted_label']
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                text = f"{label} ({conf:.2f})"
                                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                kept += 1
                                confidences.append(conf)
                        except Exception as e:
                            print(f"Error processing detection: {e}")
                    # Fallback: classify whole frame if nothing kept
                    if kept == 0 and fallback_on_empty:
                        try:
                            whole = Image.fromarray(rgb_frame)
                            result = classify_patch(whole, model, classes, transform)
                            conf = float(result.get('probability', 0.0))
                            if conf >= cls_conf:
                                label = result['predicted_label']
                                cv2.putText(frame, f"Full frame: {label} ({conf:.2f})", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                confidences.append(conf)
                        except Exception as e:
                            print(f"Fallback classification error: {e}")
                    if show_stats:
                        stats_text = f"Detections: {kept}"
                        if confidences:
                            stats_text += f" | Avg conf: {sum(confidences)/len(confidences):.2f}"
                        cv2.putText(frame, stats_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                camera_placeholder.image(frame, channels="BGR", caption="Live Trash Classification", width='stretch')
                time.sleep(0.03)  # ~30 FPS
            cap.release()
            camera_placeholder.info("Camera stopped.")
