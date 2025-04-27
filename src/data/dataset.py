import os
from torchvision import datasets, transforms
from typing import Optional, Tuple, Any, Dict
from datasets import load_dataset
import warnings
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class WasteDataset(Dataset):
    """
    Custom PyTorch Dataset for waste images organized in class folders.
    Handles corrupt images, missing/empty folders, and supports multiple file extensions.
    """
    def __init__(self, root_dir, classes, transform=None, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.samples = []
        self.extensions = extensions
        for label, cls in enumerate(classes):
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):
                files = [fname for fname in os.listdir(class_dir) if fname.lower().endswith(extensions)]
                if not files:
                    warnings.warn(f"Class folder '{cls}' is empty in '{root_dir}'.")
                for fname in files:
                    self.samples.append((os.path.join(class_dir, fname), label))
            else:
                warnings.warn(f"Class folder '{cls}' does not exist in '{root_dir}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            warnings.warn(f"Could not load image {img_path}: {e}. Returning a black image.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label

class HuggingFaceTrashNetDataset(Dataset):
    def __init__(self, hf_split, img_size=224, augment=False):
        self.hf_split = hf_split
        self.img_size = img_size
        self.augment = augment
        self.transforms = get_transforms(img_size, augment)
        self.classes = sorted(set(hf_split["label"]))
    def __len__(self):
        return len(self.hf_split)
    def __getitem__(self, idx):
        item = self.hf_split[idx]
        image = item["image"]
        label = item["label"]
        image = self.transforms(image)
        return image, label

# --- Dataset Loader Factory ---
def get_data_loaders_from_source(
    source: str,
    config: Dict[str, Any],
    classes: Optional[list] = None,
    include_test: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Unified DataLoader factory supporting multiple dataset sources.
    Supported sources:
      - 'folders': Local directory tree (default)
      - 'huggingface'/'hf': HuggingFace TrashNet
      - Custom: Extend with your own adapter class
    """
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 2)
    img_size = config.get("img_size", 224)
    augment = config.get("augment", True)
    seed = config.get("seed", None)
    if seed is not None:
        set_seed(seed)
    if source in ["huggingface", "hf"]:
        ds = load_dataset("garythung/trashnet")
        from sklearn.model_selection import train_test_split
        if "test" in ds and "validation" in ds:
            train_dataset = HuggingFaceTrashNetDataset(ds["train"], img_size=img_size, augment=augment)
            val_dataset = HuggingFaceTrashNetDataset(ds["validation"], img_size=img_size, augment=False)
            test_dataset = HuggingFaceTrashNetDataset(ds["test"], img_size=img_size, augment=False) if include_test else None
        else:
            # Split train into train/val/test (70/15/15)
            all_indices = list(range(len(ds["train"])))
            labels = ds["train"]["label"]
            train_indices, temp_indices = train_test_split(
                all_indices, test_size=0.3, random_state=seed or 42, stratify=labels
            )
            temp_labels = [labels[i] for i in temp_indices]
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=0.5, random_state=seed or 42, stratify=temp_labels
            )
            train_split = ds["train"].select(train_indices)
            val_split = ds["train"].select(val_indices)
            test_split = ds["train"].select(test_indices)
            train_dataset = HuggingFaceTrashNetDataset(train_split, img_size=img_size, augment=augment)
            val_dataset = HuggingFaceTrashNetDataset(val_split, img_size=img_size, augment=False)
            test_dataset = HuggingFaceTrashNetDataset(test_split, img_size=img_size, augment=False) if include_test else None

    elif source == "folders":
        data_dir = config["data_dir"]
        extensions = config.get("extensions", ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        train_transform = get_transforms(img_size, augment)
        val_transform = get_transforms(img_size, False)
        train_dataset = WasteDataset(os.path.join(data_dir, 'train'), classes, train_transform, extensions)
        val_dataset = WasteDataset(os.path.join(data_dir, 'val'), classes, val_transform, extensions)
        test_dataset = WasteDataset(os.path.join(data_dir, 'test'), classes, val_transform, extensions) if include_test else None
    else:
        raise ValueError(f"Unknown data source: {source}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_dataset else None
    return train_loader, val_loader, test_loader

# --- Helper: Compose transforms ---
def get_transforms(img_size, augment):
    t = [transforms.Resize((img_size, img_size))]
    if augment:
        t.append(transforms.RandomHorizontalFlip())
    t += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)

# --- Main entry point for project code ---
def get_data_loaders(config: Dict[str, Any], classes: Optional[list] = None, include_test: bool = True):
    """
    Unified DataLoader interface. Default is 'huggingface' (TrashNet via HuggingFace Datasets).
    Override by setting 'data_source' in config to 'folders' or any supported source.
    """
    source = config.get("data_source", "huggingface")
    return get_data_loaders_from_source(source, config, classes, include_test)

# --- Extending for new sources ---
# To add a new dataset source:
# 1. Create a new Dataset adapter class (subclass torch.utils.data.Dataset)
# 2. Add a new 'elif' block in get_data_loaders_from_source for your source
# 3. Document usage in README/config

def set_seed(seed):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

