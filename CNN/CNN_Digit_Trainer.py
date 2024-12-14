import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Digit Dataset
class DigitDataset(Dataset):
    def __init__(self, csv_file, num_samples=None):
        if num_samples:
            self.data = pd.read_csv(csv_file, nrows=num_samples)
        else:
            self.data = pd.read_csv(csv_file)

        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype('float32')
        image = torch.tensor(image).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Improved CNN Model
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 47)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)

# Dataset and Dataloader
train_dataset = DigitDataset(csv_file='emnist_balanced_train.csv', num_samples=400000)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Loss tracking
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training and validation functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss, correct = 0.0, 0
    for data, target in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / len(train_loader.dataset) * 100)

def validate(model, device, val_loader):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_losses.append(val_loss / len(val_loader.dataset))
    val_accuracies.append(correct / len(val_loader.dataset) * 100)
    print(f"Validation Accuracy: {correct / len(val_loader.dataset) * 100:.2f}%")

# Learning curve plot
def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.show()

# # Training loop
# for epoch in range(1, 11):  # Reduced to 10 epochs
#     train(model, device, train_loader, optimizer, epoch)
#     validate(model, device, val_loader)
#     scheduler.step()

# Save the model
# torch.save({'model_state_dict': model.state_dict()}, 'improved_cnn.pth')

# plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies)

# Confusion matrix plot
def plot_confusion_matrix(model, device, test_loader, class_names):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds.extend(output.argmax(dim=1).cpu().numpy())
            targets.extend(target.cpu().numpy())
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot()
    plt.show()


# Test on new dataset
def test_on_new_dataset(model, device, dataset_dir):
    test_dataset = DigitDataset(os.path.join(dataset_dir, 'emnist-bymerge-test.csv'))
    test_loader = DataLoader(test_dataset, batch_size=64)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds.extend(output.argmax(dim=1).cpu().numpy())
            targets.extend(target.cpu().numpy())
    print(f"New Dataset F1 Score: {f1_score(targets, preds, average='weighted'):.4f}")
    # Plot confusion matrix for new dataset
    class_names = [str(i) for i in range(47)]
    plot_confusion_matrix(model, device, test_loader, class_names)

# Instantiate the model
model = ImprovedCNN().to(device)

# Load the state dictionary into the model
checkpoint = torch.load("improved_cnn.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


test_on_new_dataset(model, device, '')