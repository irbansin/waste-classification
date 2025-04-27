import os
from torchvision import datasets, transforms
import warnings
import random
import numpy as np
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

def get_data_loaders(data_dir, classes, batch_size=32, img_size=224, augment=False, num_workers=2, seed=None, extensions=None, include_test=True):
    """
    Returns PyTorch DataLoaders for train, val, and optionally test sets.
    Applies resizing, normalization, and augmentation as needed.
    Handles random seed setting and flexible file extensions.
    """
    """Returns train, val, (optionally test) loaders. Set include_test=False to omit test set."""
    if seed is not None:
        set_seed(seed)
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = WasteDataset(os.path.join(data_dir, 'train'), classes, train_transform, extensions)
    val_dataset = WasteDataset(os.path.join(data_dir, 'val'), classes, val_transform, extensions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if include_test:
        test_dataset = WasteDataset(os.path.join(data_dir, 'test'), classes, val_transform, extensions)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

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

