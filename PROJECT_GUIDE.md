# Waste Image Classification Project Guide

This guide walks you through the structure of the Waste Image Classification project and explains how to navigate, use, and extend it step by step.

---

## 1. Project Structure Overview

```
waste-classification/
│
├── data/               # Datasets (raw, processed, splits)
├── notebooks/          # Jupyter notebooks for EDA, prototyping, visualization
├── src/                # All main source code (data, models, training logic)
│   ├── data/           # Data loaders and preprocessing
│   ├── models/         # Model architectures (CNN, ResNet, EfficientNet, etc.)
│   └── train.py        # Main training script
├── configs/            # YAML config files for experiments
├── requirements.txt    # Python dependencies
├── README.md           # Brief project overview & quickstart
├── PROJECT_GUIDE.md    # (This file) Step-by-step navigation & usage guide
└── LICENSE
```

---

## 2. Step-by-Step Navigation & Usage

### **Step 1: Set Up Your Environment**
- Create a virtual environment and install dependencies:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### **Step 2: Download and Prepare Data**
- Download datasets (e.g., TrashNet, Kaggle) and place them in `data/`.
- Organize each dataset as:
  ```
  data/<dataset_name>/train/<class_name>/*.jpg
  data/<dataset_name>/val/<class_name>/*.jpg
  data/<dataset_name>/test/<class_name>/*.jpg  # Optional but recommended
  ```
- If needed, use a notebook or script to split raw data into train/val/test.

### **Step 3: Explore the Data**
- Open Jupyter and use `notebooks/` for:
  - Visualizing images and class distributions
  - Checking for corrupt or mislabelled files
  - Trying out augmentations

### **Step 4: Configure Your Experiment**
- Edit or create a YAML config in `configs/` (e.g., `baseline.yaml`).
  - Set dataset path, class names, model type, batch size, epochs, etc.
- Example config:
  ```yaml
  data_dir: data/trashnet
  classes: [metal, plastic, paper, glass, cardboard, trash]
  model: baseline_cnn
  batch_size: 32
  img_size: 224
  epochs: 15
  lr: 0.001
  augment: true
  checkpoint_dir: outputs
  log_dir: runs/baseline_cnn
  seed: 42
  ```

### **Step 5: Train a Model**
- Run the training script with your config:
  ```bash
  python src/train.py --config configs/baseline.yaml
  ```
- Optional CLI overrides:
  ```bash
  python src/train.py --config configs/baseline.yaml --epochs 20 --batch_size 64
  ```
- Training progress, validation accuracy, and checkpoints are saved as specified in your config.

### **Step 6: Evaluate the Model**
- After training, the script will print test set accuracy (if a test set is present).
- For deeper analysis (confusion matrix, per-class metrics), use or create a notebook in `notebooks/`.

### **Step 7: Experiment with Architectures**
- Try transfer learning:
  - Change `model` in config to `resnet50` or `efficientnet_b0`.
  - You can also specify custom weights if you have them.
- Experiment with data augmentation, batch size, learning rate, and more.

### **Step 8: Semi-Supervised and Advanced Training**
- For semi-supervised learning, add unlabeled images to `data/unlabeled/`.
- Implement or extend pseudo-labeling scripts as needed (see `src/`).

### **Step 9: Extending the Project**
- **Add a new dataset:**
  - Place images in `data/`, update config, and check class names.
- **Add a new model:**
  - Add a new file in `src/models/` and update the training script if needed.
- **Add new evaluation metrics:**
  - Extend `src/train.py` or create a new script/notebook.

### **Step 10: Troubleshooting & Tips**
- If you get errors about missing classes or empty datasets, check your data structure and config.
- Check logs and warnings for corrupt images or missing folders.
- For reproducibility, always set a random seed in your config.

---

## 3. Where to Find What
- **Data loading & augmentation:** `src/data/dataset.py`
- **Model architectures:** `src/models/`
- **Training loop & logic:** `src/train.py`
- **Experiment configs:** `configs/`
- **Jupyter EDA & analysis:** `notebooks/`
- **Quickstart & summary:** `README.md`
- **This step-by-step guide:** `PROJECT_GUIDE.md`

---

## 4. Further Reading
- Each Python module is commented with docstrings and inline explanations.
- Notebooks show practical usage and visualization examples.
- For more details on datasets or model choices, see the main `README.md`.

---

## 5. Using the Web API

You can serve your trained model as a web API using FastAPI:

### **Step 1: Install API Dependencies**
All required packages are in `requirements.txt` (including fastapi, uvicorn, pillow, pyyaml).

### **Step 2: Ensure Model Weights and Config are Available**
- Place your trained model weights (e.g., `outputs/resnet50_best.pth`) and config file (e.g., `configs/baseline.yaml`) in the appropriate locations.
- You can switch between model types by editing the top of `src/api.py` (MODEL_TYPE and MODEL_PATH).

### **Step 3: Start the API Server**
```bash
uvicorn src.api:app --reload
```

### **Step 4: Test the API**
- Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI to upload images and see results interactively.
- Or use curl:
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict" -F "file=@yourimage.jpg"
  ```
- The response will include the predicted label and class probabilities.

### **Step 5: Customization**
- To use a different model, update `MODEL_TYPE` and `MODEL_PATH` at the top of `src/api.py`.
- The API will use the class names from your config file.

---

## 6. Using the Web UI

A user-friendly web interface is available for classifying images using your trained models.

### **Step 1: Start the FastAPI Backend**
Make sure your API server is running:
```bash
uvicorn src.api:app --reload
```

### **Step 2: Install UI Dependencies**
All required packages are in `requirements.txt` (including streamlit and requests):
```bash
pip install -r requirements.txt
```

### **Step 3: Launch the Streamlit Web UI**
```bash
streamlit run src/webui.py
```

### **Step 4: Use the Interface**
- Open your browser to [http://localhost:8501](http://localhost:8501)
- Upload an image and click "Classify Image".
- See the predicted class and class probabilities.

### **How it Works**
- The web UI sends your image to the FastAPI backend at `http://localhost:8000/predict`.
- The backend returns the prediction and probabilities, which are displayed in the UI.

### **Customization**
- You can change the API endpoint in `src/webui.py` if needed.
- The UI can be extended with model selection, batch upload, or result history.

---

## 7. Detecting and Classifying Multiple Waste Items

You can now process real images containing multiple types of waste in a single shot!

### **How it Works**
- The web UI has a new mode: **Detect & Classify Multiple Items**.
- Upload an image containing several waste items.
- The backend uses a YOLOv8 object detector to find and crop all visible items.
- Each cropped item is classified using your waste model.
- The UI displays:
    - The original image with bounding boxes.
    - Each cropped item, its predicted class, and probabilities.
    - Download buttons for each cropped image.

### **How to Use**
1. Start the FastAPI backend:
   ```bash
   uvicorn src.api:app --reload
   ```
2. Launch the Streamlit UI:
   ```bash
   streamlit run src/webui.py
   ```
3. Select "Detect & Classify Multiple Items" in the sidebar.
4. Upload your image and click "Detect & Classify Items".
5. View, inspect, and download results directly from the UI!

---

Happy experimenting and extending your waste classification project!
