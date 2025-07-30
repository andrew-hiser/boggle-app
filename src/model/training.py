import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
DATA_DIR = 'labeled_tiles'
BATCH_SIZE = 64
IMG_SIZE = 64
EPOCHS = 10
NUM_CLASSES = len(os.listdir(DATA_DIR))

# Map class names
class_names = sorted(os.listdir(DATA_DIR))
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for name, i in class_to_idx.items()}

# Transforms for binary (grayscale) input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # force 1 channel
    transforms.ToTensor(),                        # shape: [1, 64, 64]
    transforms.Normalize(mean=[0.5], std=[0.5])   # normalize single channel
])

# Dataset & Loader
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# CNN model for 1-channel input
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate model
model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.3f} - Acc: {acc:.3f}")

# Save model
torch.save(model.state_dict(), 'boggle_model.pth')

# Save class mapping
with open('boggle_classes.json', 'w') as f:
    json.dump(idx_to_class, f)
