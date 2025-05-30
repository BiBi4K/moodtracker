import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time

# Define ResNetEmocje model with ResNet50
class ResNetEmocje(nn.Module):
    def __init__(self, num_classes=7):
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

# Data augmentation
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

# Evaluate function for accuracy and loss per epoch
def evaluate_epoch(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    return accuracy, avg_loss

# Training function with epoch time tracking
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, save_dir='D:\\studia\\AI\\mood'):
    print("Starting training...")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_resnet_emocje_model.pth')
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing the epoch
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy, _ = evaluate_epoch(model, train_loader, criterion, device)
        val_accuracy, val_loss = evaluate_epoch(model, test_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time  # Calculate epoch duration
        
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
            torch.save(checkpoint, best_model_path, _use_new_zipfile_serialization=False)
            print(f"Saved best model at epoch {best_epoch} with Test Acc: {best_acc:.2f}%")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_time:.2f}s | Train Acc: {train_accuracy:.2f}% | Test Acc: {val_accuracy:.2f}% | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    print(f"Best model saved from epoch {best_epoch} with Test Acc: {best_acc:.2f}%")
    return best_epoch, train_accuracies, val_accuracies, train_losses, val_losses

# Evaluate function for final metrics
def evaluate(model, test_loader, class_names, criterion, device):
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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

# Plotting function for accuracy and loss
def plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, save_path='D:/studia/AI/mood/resnet_emocje_metrics_plot.png'):
    print("Plotting metrics...")
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    
    plt.subplot(2, 1, 2)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(cm, cm_accuracy, class_names, save_path='D:/studia/AI/mood/resnet_emocje_cm_plot.png'):
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
    train_dataset = datasets.ImageFolder(root='D:/studia/AI/mood/fer/train', transform=transform_train)
    test_dataset = datasets.ImageFolder(root='D:/studia/AI/mood/fer/test', transform=transform_test)
    print(f"Datasets loaded: {len(train_dataset)} train, {len(test_dataset)} test images.")
    
    print("Computing class weights...")
    targets = [label for _, label in train_dataset]
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * 7.0
    print(f"Class weights: {class_weights.tolist()}")
    
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    print("DataLoaders created with num_workers=4.")
    
    print("Initializing model...")
    model = ResNetEmocje(num_classes=7).to(device)
    print("Model initialized successfully with pretrained ResNet50 weights.")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.005)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.7, min_lr=5e-5)
    
    print("Starting main training...")
    best_epoch, train_accuracies, val_accuracies, train_losses, val_losses = train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=150, save_dir='D:\\studia\\AI\\mood')
    
    print("Loading best model for evaluation...")
    checkpoint = torch.load('D:/studia/AI/mood/best_resnet_emocje_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with Test Acc: {checkpoint['test_accuracy']:.2f}%")
    
    class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    cm, cm_accuracy = evaluate(model, test_loader, class_names, criterion, device)
    
    plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses)
    plot_confusion_matrix(cm, cm_accuracy, class_names)
    print("Plots saved at D:/studia/AI/mood/resnet_emocje_metrics_plot.png and D:/studia/AI/mood/resnet_emocje_cm_plot.png")