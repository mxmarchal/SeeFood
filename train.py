import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Data preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load data
train_data = datasets.ImageFolder('./dataset/train', data_transforms['train'])
val_data = datasets.ImageFolder('./dataset/test', data_transforms['val'])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Define model
model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Move the model to the GPU (Apple Metal)
if not torch.has_mps:
    print("Metal backend not available")
    exit()
device = torch.device("mps")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Training loop
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()

    print(f'Epoch {epoch + 1}, Validation Accuracy: {total_correct / len(val_data)}')

# Save the model
model_save_path = "hotdog_classifier.pth"
torch.save(model.cpu().state_dict(), model_save_path) # Move the model back to the CPU before saving so it can be loaded on other devices
