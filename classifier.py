import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Prevent getting error because of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Define device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Define constants
    BATCH_SIZE = 64
    EPOCHS = 10

    # Implement custom dataset
    class CatsVsDogsDataset(Dataset):
        def __init__(self, root_dir, transforms=None):
            super().__init__()
            self.root_dir = root_dir
            self.image_paths = []
            self.labels = []
            self.transforms = transforms

            valid_exts = (".jpg", ".jpeg", ".png")

            for label_name, label_id in [('Cat', 0), ('Dog', 1)]:
                class_dir = os.path.join(root_dir, label_name)
                for fname in os.listdir(class_dir):
                    # Check files are valid
                    if not fname.lower().endswith(valid_exts):
                        continue
                    if fname.startswith(".") or fname.lower() == "thumbs.db":
                        continue

                    fpath = os.path.join(class_dir, fname)

                    # Exclude unwanted data
                    try:
                        with Image.open(fpath) as im:
                            im.verify()
                    except Exception:
                        continue

                    self.image_paths.append(fpath)
                    self.labels.append(label_id)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            img = Image.open(path).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return img, self.labels[idx]
        
    # Define transforms
    data_transforms = transforms.Compose([
        # Resizing and centercropping images so we prevent resizing error to be occured
        transforms.Resize(256),
        transforms.CenterCrop(224),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Mean and std values for resnet18 architecture (Better normalization)
    ])
        
    # Split dataset to train and test parts
    root = "/home/cagan/tl-catsvdogs-pytorch/archive/PetImages"
    full_ds = CatsVsDogsDataset(root_dir=root, transforms=data_transforms)

    labels = np.array(full_ds.labels)
    rng = np.random.default_rng(42) # Random seed

    cat_idx = np.where(labels == 0)[0]
    dog_idx = np.where(labels == 1)[0]
    rng.shuffle(cat_idx)
    rng.shuffle(dog_idx)

    val_ratio = 0.2
    n_cat_val = int(len(cat_idx) * val_ratio)
    n_dog_val = int(len(dog_idx) * val_ratio)

    val_idx = np.concatenate([cat_idx[:n_cat_val], dog_idx[:n_dog_val]])
    train_idx = np.concatenate([cat_idx[n_cat_val:], dog_idx[n_dog_val:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())

    # Load data using DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Define model, loss function and optimizer
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2) # Cat, Dog
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch: {epoch}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
            
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_accuracy}")

    # Save the model
    torch.save(model.state_dict(), 'cvd_classifier.pth')