import random

import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder

from transform_images import(
    jpeg_compression,
    gaussian_blur,
    resize_image,
    gaussian_noise,
    colour_jitter,
    center_crop,
    apply_random_transform
)

from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_sid = load_dataset("saberzl/SID_Set", split="train", streaming=True)
val_sid = load_dataset("saberzl/SID_Set", split="validation", streaming=True)

class SIDStreamingDataset(IterableDataset):
    def __init__(self, hf_dataset, transform=None, augmentation=None, max_per_class=1000):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.augmentation=augmentation
        self.max_per_class = max_per_class
    def __iter__(self):
        class_counts = {0: 0, 1: 0}
        for example in self.hf_dataset:
            original_label = example["label"]
            ## 0: real, 1: AI
            if original_label == 0:
                label = 0
            elif original_label in [1, 2]:
                label = 1
            else:
                continue
            if class_counts[label] >= self.max_per_class:
                continue

            image = example["image"].convert("RGB")
            if self.augmentation:
              image = self.augmentation(image)
            if self.transform:
                image = self.transform(image)
            class_counts[label] += 1
            yield image, label
            if all(count >= self.max_per_class for count in class_counts.values()):
                break

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

sid_train = SIDStreamingDataset(hf_dataset = train_sid, transform=transform, augmentation=apply_random_transform,max_per_class=1000)
sid_val = SIDStreamingDataset(hf_dataset = val_sid, transform=transform, augmentation=None, max_per_class=1000)

train_cifake = ImageFolder("/content/CIFAKE/train")
test_cifake = ImageFolder("/content/CIFAKE/test")

random.seed(42)

fake_indices = [i for i, (_, label) in enumerate(train_cifake.samples) if label == train_cifake.class_to_idx["FAKE"]]
real_indices = [i for i, (_, label) in enumerate(train_cifake.samples) if label == train_cifake.class_to_idx["REAL"]]

cifake_fake_indices = random.sample(fake_indices, 1000)
cifake_real_indices = random.sample(real_indices, 1000)
cifake_indices = (cifake_fake_indices + cifake_real_indices)

random.shuffle(cifake_indices)

class CIFAKEDataset(Dataset):
    def __init__(self, cifake_dataset, indices, transform=None, augmentation=None):
        self.cifake_dataset = cifake_dataset
        self.indices = indices
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        if self.indices is None:
            return len(self.cifake_dataset)
        return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is None:
            original_idx = idx
        else:
            original_idx = self.indices[idx]
        image, label = self.cifake_dataset[original_idx]
        if label == 1:
            label = 0
        elif label == 0:
            label = 1
        else:
            pass

        image = image.convert("RGB")
        if self.augmentation:
          image = self.augmentation(image)
        if self.transform:
            image = self.transform(image)
        return image, label

cifake_train = CIFAKEDataset(train_cifake, cifake_indices, transform=transform, augmentation=apply_random_transform)
cifake_val = CIFAKEDataset(test_cifake, indices=None, transform=transform, augmentation=None)

class CombinedDataset(IterableDataset):
    def __init__(self, sid_dataset, cifake_dataset):
        self.sid_dataset = sid_dataset
        self.cifake_dataset = cifake_dataset
    def __iter__(self):
        sid_iter = iter(self.sid_dataset)
        cifake_iter = iter(self.cifake_dataset)
        sid_finished = False
        cifake_finished = False
        while not(sid_finished and cifake_finished):
            if not sid_finished:
                try:
                    yield next(sid_iter)
                except StopIteration:
                    sid_finished = True

            if not cifake_finished:
                try:
                    yield next(cifake_iter)
                except StopIteration:
                    cifake_finished = True

train_combined = CombinedDataset(sid_train, cifake_train)
train_loader = DataLoader(train_combined, batch_size=32, num_workers=0)
sid_val_loader = DataLoader(sid_val, batch_size=32, num_workers=0)
cifake_val_loader = DataLoader(cifake_val, batch_size=32, num_workers=0)

model = resnet18(weights=None)
model.fc = nn.Linear(512, 2)
model.load_state_dict(
    torch.load(
        "/content/resnet18_1000perclass_20epochs_combinedataset.pth",
        map_location=device
    )
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    num_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    average_loss = running_loss / num_samples

    model.eval()

    sid_correct = 0
    sid_total = 0

    with torch.no_grad():
        for images, labels in sid_val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            sid_correct += (predictions == labels).sum().item()
            sid_total += labels.size(0)
    sid_accuracy = (sid_correct / sid_total)

    cifake_correct = 0
    cifake_total = 0

    with torch.no_grad():
        for images, labels in cifake_val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            cifake_correct += (predictions == labels).sum().item()
            cifake_total += labels.size(0)
    cifake_accuracy = (cifake_correct / cifake_total)


        ## what about model accuracy?
    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Training Loss: {average_loss:.4f} | "
        f"SID Accuracy: {sid_accuracy:.4f} | "
        f"CIFAKE Accuracy: {cifake_accuracy:.4f}"
    )
