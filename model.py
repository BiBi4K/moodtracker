import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout2d(0.5)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout2d(0.5)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(512)
        self.dropout5 = nn.Dropout2d(0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        dummy_input = torch.zeros(1, 3, 48, 48)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.dropout3(self.relu(self.batch_norm3(self.conv3(x)))))
        x = self.pool(self.dropout4(self.relu(self.batch_norm4(self.conv4(x)))))
        x = self.pool(self.dropout5(self.relu(self.batch_norm5(self.conv5(x)))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(self.relu(self.fc1(x)))
        x = self.dropout_fc(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(48),
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root='D:/studia/AI/mood/fer/train', transform=transform_train)
test_dataset = datasets.ImageFolder(root='D:/studia/AI/mood/fer/test', transform=transform_test)

indices = list(range(len(train_dataset)))
labels = [train_dataset.targets[i] for i in indices]
train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=6, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(train_dataset, batch_size=64, sampler=val_sampler, num_workers=6, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True, persistent_workers=True)

targets = [label for _, label in train_dataset]
class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)

class_weights[1] *= 2  # Disgust
class_weights[2] *= 2  # Fear
class_weights[5] *= 2  # Surprise
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
fine_tuning_scheduler = CosineAnnealingLR(optimizer, T_max=50)
train_accuracies = []
val_accuracies = []

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=150, save_dir='D:\\studia\\AI\\mood'):
    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        val_accuracy = calculate_accuracy(model, val_loader)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(running_loss)
        elapsed_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] finished in {elapsed_time:.2f}s | "
              f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | "
              f"Loss: {running_loss/len(train_loader):.4f}")
        
        if val_accuracy >= best_acc + 0.1:
            best_acc = val_accuracy
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': running_loss / len(train_loader),
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model with Val Acc: {best_acc:.2f}% at {best_model_path}")

def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Test Accuracy: {acc * 100:.2f}%, F1 Score: {f1:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    cm_accuracy = cm / cm.sum(axis=1)[:, np.newaxis]
    return cm, cm_accuracy

def plot_accuracy():
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, cm_accuracy, class_names):
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_accuracy, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with Class-wise Accuracy')
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=150)
    fine_tuning_scheduler = CosineAnnealingLR(optimizer, T_max=10)
    train(model, train_loader, val_loader, criterion, optimizer, fine_tuning_scheduler, num_epochs=50)
    cm, cm_accuracy = evaluate(model, test_loader)
    end_time = time.time()
    print(f"Training and Evaluation Time: {end_time - start_time:.2f} seconds")
    plot_accuracy()
    class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    plot_confusion_matrix(cm, cm_accuracy, class_names)
