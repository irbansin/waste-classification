import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.data.dataset import get_data_loaders
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import get_resnet_model
from src.models.efficientnet import get_efficientnet_model
import yaml

MODEL_MAP = {
    'baseline_cnn': BaselineCNN,
    'resnet50': get_resnet_model,
    'efficientnet_b0': lambda num_classes, pretrained: get_efficientnet_model(num_classes, 'efficientnet_b0', pretrained)
}

def train(config):
    """
    Main training loop for image classification models.
    Loads data, builds model, trains with early stopping and learning rate scheduling,
    and evaluates on test set if available.
    Args:
        config (dict): configuration dictionary (can be loaded from YAML)
    """
    import warnings
    from src.data.dataset import set_seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    classes = config['classes']
    # Get data loaders (train, val, test)
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            config['data_dir'],
            classes,
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            augment=config['augment'],
            num_workers=config.get('num_workers', 2),
            seed=seed,
            include_test=True
        )
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Train or validation dataset is empty. Check your data directory and class names.")
    model_type = config['model']
    pretrained = config.get('pretrained', False)
    num_classes = len(classes)
    # Model selection
    try:
        if model_type == 'baseline_cnn':
            model = BaselineCNN(num_classes=num_classes)
        elif model_type == 'resnet50':
            model = get_resnet_model(num_classes=num_classes, pretrained=pretrained)
        elif model_type == 'efficientnet_b0':
            model = get_efficientnet_model(num_classes=num_classes, model_name='efficientnet_b0', pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    writer = SummaryWriter(log_dir=config.get('log_dir', './runs'))
    best_acc = 0.0
    patience = config.get('early_stopping_patience', 7)
    epochs_no_improve = 0
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total if total > 0 else 0
        val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        scheduler.step(val_acc)
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            os.makedirs(config['checkpoint_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], f"{model_type}_best.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    print(f"Best validation accuracy: {best_acc:.4f}")
    writer.close()
    # Test set evaluation
    if test_loader and len(test_loader.dataset) > 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total if total > 0 else 0
        print(f"Test accuracy: {test_acc:.4f}")
    else:
        print("No test set found or test set is empty.")

def main():
    """
    Command-line entry point.
    Loads config, applies CLI overrides, and starts training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--epochs', type=int, help='Override epochs in config')
    parser.add_argument('--batch_size', type=int, help='Override batch size in config')
    parser.add_argument('--lr', type=float, help='Override learning rate in config')
    parser.add_argument('--num_workers', type=int, help='Override num_workers in config')
    parser.add_argument('--seed', type=int, help='Override random seed in config')
    args = parser.parse_args()
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Could not load config file: {e}")
    # Override config with CLI args if provided
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.seed is not None:
        config['seed'] = args.seed
    train(config)


if __name__ == '__main__':
    main()
