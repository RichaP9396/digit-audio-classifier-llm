import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load features
data = np.load("data/fsdd_features.npz")
X = data["X"]  # shape: (num_samples, n_mfcc, frames)
y = data["y"]

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 13, 32)
y = torch.tensor(y, dtype=torch.long)

# Dataset and split
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# CNN model with dynamic flattening
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        
        # Compute flattened size dynamically
        dummy = torch.zeros(1,1,13,32)
        dummy = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy)))))
        flattened_size = dummy.numel()
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            outputs = model(val_X)
            _, predicted = torch.max(outputs.data, 1)
            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()
    val_acc = correct / total * 100
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), "digit_cnn.pth")
print("Training complete! Model saved as digit_cnn.pth")