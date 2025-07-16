import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18
from utils import plot_loss
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load dataset
dataset = datasets.ImageFolder('data', transform=transform)

# Split dataset into 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Initialize model
# Load pretrained ResNet18
model = resnet18(weights='IMAGENET1K_V1')  # Requires torchvision>=0.13
model.fc = nn.Linear(model.fc.in_features, 2)  # Replace classifier head
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_history = []
train_acc_history = []
val_acc_history = []

# Training loop
for epoch in range(20):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/20]", leave=False)
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Training accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    loss_history.append(avg_loss)
    train_acc_history.append(train_accuracy)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_accuracy = 100 * val_correct / val_total
    val_acc_history.append(val_accuracy)

    print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f} | Train Acc = {train_accuracy:.2f}% | Val Acc = {val_accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), 'models/resnet.pth')
print("✅ Model saved to models/resnet.pth")

# Evaluate on validation set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate report
print("\n🧾 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


# Plot training loss curve
# plot_loss(loss_history)

# Optional: plot validation accuracy (you can add plot_accuracy to utils.py)
# plot_accuracy(val_acc_history)
