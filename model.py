import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import time
import os
import pandas as pd
from PIL import Image

class FERPlusDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_file, header=None)
        self.class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        votes = self.labels_df.iloc[idx, 2:10].values.astype(int)
        label = np.argmax(votes)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ResNetEmocje(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetEmocje, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.maxpool = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer1.parameters():
            param.requires_grad = True
        for param in self.model.layer2.parameters():
            param.requires_grad = True
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.7, scale=(0.02, 0.4))
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(48),
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def evaluate_epoch(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    return accuracy, avg_loss

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs, save_dir='D:\\studia\\AI\\mood', start_epoch=0, all_metrics=None):
    print("Starting training...")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'model.pth')

    if all_metrics is None:
        all_metrics = {
            'train_accuracies': [],
            'val_accuracies': [],
            'train_losses': [],
            'val_losses': []
        }
    
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy, _ = evaluate_epoch(model, train_loader, criterion, device)
        val_accuracy, val_loss = evaluate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_epoch = epoch + 1
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': val_accuracy
            }
            torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
            print(f"Saved best model at epoch {best_epoch} with Val Acc: {best_acc:.2f}%")
        
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {elapsed_time:.2f}s | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        all_metrics['train_accuracies'].append(train_accuracy)
        all_metrics['val_accuracies'].append(val_accuracy)
        all_metrics['train_losses'].append(train_loss)
        all_metrics['val_losses'].append(val_loss)
    
    print(f"Best model saved from epoch {best_epoch} with Val Acc: {best_acc:.2f}%")
    return best_epoch, all_metrics

def evaluate(model, test_loader, class_names, criterion, device):
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = running_loss / len(test_loader)
    print(f"Test Accuracy: {acc * 100:.2f}%, F1 Score: {f1:.4f}, Test Loss: {avg_loss:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_accuracy = np.divide(cm, cm.sum(axis=1)[:, np.newaxis], out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1)[:, np.newaxis] != 0)
    return cm, cm_accuracy

def plot_metrics(all_metrics, main_training_epochs, save_path='D:/studia/AI/mood/metrics_plot.png'):
    print("Plotting metrics...")
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 1, 1)
    plt.plot(all_metrics['train_accuracies'], label='Training Accuracy', color='blue')
    plt.plot(all_metrics['val_accuracies'], label='Validation Accuracy', color='orange')
    plt.axvline(x=main_training_epochs, color='red', linestyle='--', label='Main Training End')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    
    plt.subplot(2, 1, 2)
    plt.plot(all_metrics['train_losses'], label='Training Loss', color='blue')
    plt.plot(all_metrics['val_losses'], label='Validation Loss', color='orange')
    plt.axvline(x=main_training_epochs, color='red', linestyle='--', label='Main Training End')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, cm_accuracy, class_names, save_path='D:/studia/AI/mood/cm_plot.png'):
    print("Plotting confusion matrix...")
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_accuracy, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with Class-wise Accuracy')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_dataset = FERPlusDataset(
        image_dir='D:/studia/AI/mood/FER2013Plus/Images/FER2013Train',
        label_file='D:/studia/AI/mood/FER2013Plus/Labels/FER2013Train/label.csv',
        transform=transform_train
    )
    val_dataset = FERPlusDataset(
        image_dir='D:/studia/AI/mood/FER2013Plus/Images/FER2013Valid',
        label_file='D:/studia/AI/mood/FER2013Plus/Labels/FER2013Valid/label.csv',
        transform=transform_test
    )
    test_dataset = FERPlusDataset(
        image_dir='D:/studia/AI/mood/FER2013Plus/Images/FER2013Test',
        label_file='D:/studia/AI/mood/FER2013Plus/Labels/FER2013Test/label.csv',
        transform=transform_test
    )
    print(f"Datasets loaded: {len(train_dataset)} train, {len(val_dataset)} valid, {len(test_dataset)} test images.")
    
    print("Computing class weights...")
    targets = [train_dataset[idx][1] for idx in range(len(train_dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * 8.0
    print(f"Class weights: {class_weights.tolist()}")
    
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=6, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=6, pin_memory=True, persistent_workers=True)
    print("DataLoaders created with num_workers=6.")
    
    print("Initializing model...")
    model = ResNetEmocje(num_classes=8).to(device)
    print("Model initialized successfully with pretrained ResNet50 weights.")
    
    print("Warming up CUDA...")
    with torch.no_grad():
        model.eval()
        dummy_input = torch.randn(1, 1, 48, 48).to(device)
        model(dummy_input)
    print("CUDA warm-up complete.")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.005)
    scaler = GradScaler()
    
    start_time = time.time()
    print("Starting main training...")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.7, min_lr=5e-5)
    main_training_epochs = 150
    best_epoch, all_metrics = train(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=main_training_epochs, save_dir='D:\\moodtracker')
    
    #print("Starting fine-tuning...")
    #for param in model.model.parameters():
    #    param.requires_grad = True
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-3)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.7, min_lr=5e-5)
    #best_epoch, all_metrics = train(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100, save_dir='D:\\moodtracker', start_epoch=main_training_epochs, all_metrics=all_metrics)
    
    print("Loading best model for evaluation...")
    checkpoint = torch.load('D:/studia/AI/mood/model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with Val Acc: {checkpoint['test_accuracy']:.2f}%")
    
    class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    cm, cm_accuracy = evaluate(model, test_loader, class_names, criterion, device)
    
    end_time = time.time()
    print(f"Training and Evaluation Time: {end_time - start_time:.2f} seconds")
    
    plot_metrics(all_metrics, main_training_epochs)
    plot_confusion_matrix(cm, cm_accuracy, class_names)
    print("Plots saved at D:/studia/AI/mood/metrics_plot.png and D:/studia/AI/mood/cm_plot.png")
