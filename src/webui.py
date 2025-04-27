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
mode = st.sidebar.radio("Choose Mode", ["Classify Single Waste Item", "Detect & Classify Multiple Items", "Detect & Classify from Video"])

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

elif mode == "Detect & Classify from Video":
    st.header("Detect & Classify Trash from Video")
    uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"], key="video")
    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("Detect & Classify Trash in Video"):
            with st.spinner("Processing video and classifying frames..."):
                files = {'file': (uploaded_video.name, uploaded_video, uploaded_video.type)}
                try:
                    response = requests.post(f"{API_URL}/predict_video", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Video processed. See results below.")
                        st.subheader("Aggregated Class Counts Across Video")
                        st.json(result.get("class_counts", {}))
                        st.subheader("Detected Trash Items in Sampled Frames")
                        frame_results = result.get("frame_results", [])
                        if frame_results:
                            max_frames = min(10, len(frame_results))
                            for fr in frame_results[:max_frames]:
                                st.markdown(f"**Frame {fr['frame']}**")
                                patches = fr.get('patches', [])
                                if patches:
                                    for i, patch in enumerate(patches):
                                        st.markdown(f"**Patch {i+1}: {patch['predicted_label']}** (prob: {patch['probability']:.2f})")
                                        coords = patch.get('coords', [])
                                        st.write(f"Coords: {coords}")
                                        patch_img = Image.open(io.BytesIO(base64.b64decode(patch['patch_b64'])))
                                        st.image(patch_img, caption=f"Patch {i+1}", width=120)
                                    st.markdown("---")
                                else:
                                    st.info("No patches detected in this frame.")
                            if len(frame_results) > max_frames:
                                st.info(f"...and {len(frame_results) - max_frames} more frames processed.")
                        else:
                            st.info("No patch detections found in the video frames.")
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
    else:
        st.info("Please upload a video file to begin.")
