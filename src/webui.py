"""
Streamlit Web UI for Waste Image Classification
Allows users to upload an image, select a model, and view predictions interactively.
"""
import streamlit as st
import requests
from PIL import Image
import io
import base64

API_URL = "http://localhost:8001"

st.set_page_config(page_title="Waste Image Classifier", layout="centered")
st.title("Waste Image Classification Web UI")

# Tabs for single and multi-item detection/classification
mode = st.sidebar.radio("Choose Mode", [
    "Classify Single Waste Item",
    "Detect & Classify Multiple Items",
    "Live Camera Trash Classification"
])

if mode == "Classify Single Waste Item":
    st.header("Classify a Single Waste Item")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("")
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                img_bytes.seek(0)
                files = {'file': (uploaded_file.name, img_bytes, uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/predict", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction: {result['predicted_label']} (probability: {result['probability']:.2f})")
                        st.write("Class Probabilities:")
                        st.json(result['class_probs'])
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
    else:
        st.info("Please upload an image file to begin.")

elif mode == "Detect & Classify Multiple Items":
    st.header("Detect & Classify Multiple Waste Items")
    uploaded_file = st.file_uploader("Upload an image with multiple waste items...", type=["jpg", "jpeg", "png"], key="multi")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Detect & Classify Items"):
            with st.spinner("Detecting and Classifying..."):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                img_bytes.seek(0)
                files = {'file': (uploaded_file.name, img_bytes, uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/detect_and_classify", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        # Show image with bounding boxes
                        img_with_boxes_b64 = result['image_with_boxes_b64']
                        img_with_boxes = Image.open(io.BytesIO(base64.b64decode(img_with_boxes_b64)))
                        st.image(img_with_boxes, caption="Detected Items", use_container_width=True)
                        st.subheader("Detected Waste Items:")
                        for i, det in enumerate(result['detections']):
                            st.markdown(f"**Item {i+1}: {det['predicted_label']}** (prob: {det['probability']:.2f})")
                            crop_img = Image.open(io.BytesIO(base64.b64decode(det['crop_b64'])))
                            st.image(crop_img, caption=f"Cropped Item {i+1}", width=200)
                            st.write("Class Probabilities:")
                            st.json(det['class_probs'])
                            # Download button for crop
                            crop_bytes = io.BytesIO()
                            crop_img.save(crop_bytes, format='JPEG')
                            st.download_button(
                                label=f"Download Cropped Item {i+1}",
                                data=crop_bytes.getvalue(),
                                file_name=f"waste_item_{i+1}.jpg",
                                mime="image/jpeg"
                            )
                            st.markdown("---")
                        if not result['detections']:
                            st.warning("No waste items detected with sufficient confidence.")
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
    else:
        st.info("Please upload an image file to begin.")

elif mode == "Live Camera Trash Classification":
    st.header("Live Camera Trash Classification")
    st.markdown("""
    This mode processes your camera feed in real time, detects and classifies trash,
    and overlays the class and hierarchy on the video. Use this as the decision engine for a robot.
    """)
    st.warning("Live video feed will open in a separate window. Close it or press 'q' to stop.")
    if st.button("Start Live Camera Processing"):
        import subprocess
        st.info("Launching real-time decision engine...")
        # Launch the external script as a subprocess
        proc = subprocess.Popen(["python", "realtime_decision_engine.py"])
        st.success("Real-time camera processing started. Check the new window.")
        st.info("To stop, close the camera window or press 'q'.")
