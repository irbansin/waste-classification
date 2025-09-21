# Data Directory for Waste Classification

This folder contains raw and processed datasets for training and evaluating waste classification models.

## Recommended Datasets

1. **TrashNet** (Stanford)
   - [TrashNet on GitHub](https://github.com/GaryThung/trashnet)
   - ~2,527 images, 6 classes: glass, paper, cardboard, plastic, metal, trash
   - Download and extract into `data/trashnet/` so you have:
     ```
     data/trashnet/
         glass/
         paper/
         cardboard/
         plastic/
         metal/
         trash/
     ```

2. **Garbage Classification (12-Class Dataset)**
   - [Papers With Code Link](https://paperswithcode.com/dataset/garbage-classification-dataset)
   - ~15,150 images, 12 classes
   - Download and extract as `data/garbage_12/`

3. **Waste Classification Kaggle**
   - [Kaggle Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data)
   - ~25,000 images, 2–3 classes
   - Download and extract as `data/waste_kaggle/`

4. **OpenLitterMap**
   - [OpenLitterMap](https://openlittermap.com/)
   - 100k+ images, multi-label
   - See their documentation for download instructions (may require API or scraping)

## Data Preparation
- For each dataset, split into `train/` and `val/` folders for supervised training.
- Ensure folder structure is:
  ```
  data/<dataset_name>/train/<class_name>/*.jpg
  data/<dataset_name>/val/<class_name>/*.jpg
  ```
- You can use scripts in `notebooks/` or write your own to split data.

## Notes
- Always check license terms for each dataset.
- For semi-supervised learning, place unlabeled images in `data/unlabeled/`.
- If you add a new dataset, update this README and the data loader code.
