# Waste Image Classification Project

## Project Overview
This project builds a deep learning system to classify waste images into multiple categories for improved recycling and waste management. It supports hierarchical classification: distinguishing Organic vs Inorganic waste, and further classifying Inorganic waste into Medical, Electronic, or General (Metal, Plastic, Paper).

## Features
- Modular data pipeline for multiple datasets (TrashNet, Garbage Classification, etc.)
- Baseline CNN and advanced models (ResNet, EfficientNet, Vision Transformer)
- Hierarchical and flat classification options
- Semi-supervised learning with pseudo-labeling
- Data augmentation and class imbalance handling
- Per-class and hierarchical evaluation metrics

## Project Structure
```
waste-classification/
│
├── data/                   # Place for raw and processed datasets
│   └── README.md           # Instructions for downloading datasets
├── notebooks/              # Jupyter notebooks for EDA, prototyping, and visualization
├── src/                    # Source code for data, models, training, etc.
├── configs/                # YAML/JSON configs for experiments
├── scripts/                # Downloading data, running experiments
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
└── LICENSE
```

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data
See `data/README.md` for dataset links and organization. Example for TrashNet:
```bash
bash scripts/download_trashnet.sh
```

### 3. Train Baseline Model
```bash
python src/train.py --config configs/baseline.yaml
```

### 4. Evaluate Model
```bash
python src/evaluate.py --checkpoint outputs/baseline_cnn.pth
```

### 5. Experiment with Transfer Learning
Edit configs and run:
```bash
python src/train.py --config configs/resnet.yaml
```

### 6. Semi-Supervised Learning
```bash
python src/pseudo_label.py --config configs/pseudo_label.yaml
```

## Datasets Supported
- TrashNet (~2.5k, 6 classes)
- Garbage Classification (12-class, ~15k)
- Waste Classification Kaggle (2–3 classes, ~25k)
- OpenLitterMap (large-scale, multi-label)

## Model Architectures
- Baseline CNN (from scratch)
- Transfer Learning (ResNet, EfficientNet)
- Vision Transformer (ViT)
- Hierarchical (two-stage) or flat classifier

## Semi-Supervised Learning
- Pseudo-labeling pipeline for unlabeled images
- Consistency training (optional)

## Evaluation
- Per-class accuracy, precision, recall
- Confusion matrix
- Hierarchical accuracy

## How to Extend
- Add new datasets: update `src/data/dataset.py`
- Add new models: extend `src/models/`
- Add new experiments: create config files in `configs/`

## References
- [TrashNet Dataset](https://github.com/GaryThung/trashnet)
- [Garbage Classification Dataset](https://paperswithcode.com/dataset/garbage-classification-dataset)
- [OpenLitterMap](https://openlittermap.com/)

---

For detailed documentation, see comments in each module and the notebooks for examples.
