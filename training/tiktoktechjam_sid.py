## not final code
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import confusion_matrix, classification_report

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
  print("GPU:", torch.cuda.get_device_name(0))

train_hf = load_dataset(
    "saberzl/SID_Set",
    split="train",
    streaming=True
)

val_hf = load_dataset(
    "saberzl/SID_Set",
    split="validation",
    streaming=True
)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

model.fc = nn.Linear(512, 2)

model = model.to(device)

transform = weights.transforms()

class SIDStreamingDataset(IterableDataset):

  def __init__(
      self,
      hf_dataset,
      transform=None,
      max_per_class=5000
  ):
    self.hf_dataset = hf_dataset
    self.transform = transform
    self.max_per_class = max_per_class

  def __iter__(self):
    class_counts = {
        0: 0,
        1: 0
    }

    for example in self.hf_dataset:

      label = example["label"]

      if label == 0:
        pass
      elif label in [1, 2]:
        label = 1

      if class_counts[label] >= self.max_per_class:
        continue
      
      image = example["image"]

      image = image.convert("RGB")

      if self.transform:
        image = self.transform(image)
      
      class_counts[label] += 1

      yield image, label

      if all(
          count >= self.max_per_class
          for count in class_counts.values()
      ):
        break

train_dataset = SIDStreamingDataset(
    train_hf,
    transform=transform,
    max_per_class=10000
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=0
)

val_dataset = SIDStreamingDataset(
    val_hf,
    transform=transform,
    max_per_class=500
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=0
)

images, labels = next(iter(train_loader))

print("Images:", images.shape)
print("Labels:", labels.shape)
print("Labels:", labels)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

num_epochs = 10

for epoch in range(num_epochs):
  model.train()

  running_loss = 0.0
  num_batches = 0

  for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    running_loss += loss.item()
    num_batches += 1
  
  average_loss = running_loss / num_batches

  model.eval()

  correct = 0
  total = 0

  all_predictions = []
  all_labels = []

  with torch.no_grad():
    for images, labels in val_loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)

      predictions = torch.argmax(
          outputs,
          dim=1
      )

      correct += (
          predictions == labels
      ).sum().item()

      total += labels.size(0)

      all_predictions.extend(
          predictions.cpu().numpy()
      )

      all_labels.extend(
          labels.cpu().numpy()
      )

  validation_accuracy = correct / total

  print(
    f"Epoch {epoch + 1}/{num_epochs} | "
    f"Training Loss: {average_loss:.4f} | "
    f"Validation Accuracy: {validation_accuracy:.4f}"
    )

print("\nFinal Classification Report:"
)
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Real",
            "AI"
        ]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)
