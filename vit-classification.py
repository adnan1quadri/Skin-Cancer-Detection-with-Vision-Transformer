# Import libraries
import torch 

from torch import nn 

import torch.optim as optim  
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset,DataLoader
from torch import Tensor
# ----------------------

import os
import pandas as pd
from PIL import Image
import h5py
import tqdm
import io
from io import BytesIO
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import DeiTForImageClassification
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

torch.__version__

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

rain_matadata = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv", low_memory=False)
train_matadata.head()

test_matadata = pd.read_csv("/kaggle/input/isic-2024-challenge/test-metadata.csv", low_memory=False)
test_matadata.head()

# Data Augmentation
import torchvision.transforms as transforms

train_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),  # Ensure all images are resized to 224x224 first
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=15),  # Randomly rotate images
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, etc.
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformations
    transforms.RandomGrayscale(p=0.1),  # Randomly convert images to grayscale
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

valid_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),  # Ensure all images are resized to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),  # Ensure all images are resized to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Fetch the Dataset - prepare dataloader 
class ImageLoader(Dataset):
    def __init__(self, df, file_hdf, transform=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        image = Image.open(BytesIO(self.fp_hdf[isic_id][()]))
        target = self.targets[index]
        
        if self.transform:
            return (self.transform(image), target)
        else:
            return (image, target)

# Train/test split--------
def stratified_train_valid_test_split(df, valid_size=0.2, test_size=0.15, random_state=None):
    # First split: train + valid and test
    train_valid_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['target'], random_state=random_state
    )
    
    # Second split: train and valid
    train_df, valid_df = train_test_split(
        train_valid_df, test_size=valid_size / (1 - test_size), stratify=train_valid_df['target'], random_state=random_state
    )
    
    return train_df, valid_df, test_df

train_df, valid_df, test_df = stratified_train_valid_test_split(train_matadata, valid_size=0.2, test_size=0.15, random_state=42)

file_hdf = "/kaggle/input/isic-2024-challenge/train-image.hdf5"
train_dataset = ImageLoader(train_df, file_hdf, transform=train_transforms)
valid_dataset = ImageLoader(valid_df, file_hdf, transform=valid_transforms)
test_dataset = ImageLoader(test_df, file_hdf, transform=test_transforms)

# Alternatively, if the labels are numeric (0, 1, 2, ...), you can use np.bincount for speed:
from collections import Counter
import numpy as np

train_label_counts_np = np.bincount(train_dataset.targets)
valid_label_counts_np = np.bincount(valid_dataset.targets)
test_label_counts_np = np.bincount(test_dataset.targets)

# If using np.bincount
print("Label counts in the train dataset (using np.bincount):", train_label_counts_np)
print("Label counts in the valid dataset (using np.bincount):", valid_label_counts_np)
print("Label counts in the test dataset (using np.bincount):", test_label_counts_np)

# Ratio of label 0 to label 1 

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers = 2)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers = 2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers = 2)

# Test data loader for kaggle challenge submission
class ImageLoader_test(Dataset):
    def __init__(self, df, file_hdf, transform=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
#         self.targets = df['target'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        image = Image.open(BytesIO(self.fp_hdf[isic_id][()]))
#         target = self.targets[index]
        
        if self.transform:
            return self.transform(image)
        else:
            return image

file_hdf_test = "/kaggle/input/isic-2024-challenge/test-image.hdf5"
test_actual = ImageLoader_test(test_matadata, file_hdf_test, transform=train_transforms)

test_loader_actual = DataLoader(test_actual, batch_size=128, shuffle=False, num_workers = 4)

# Plot random images
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: list[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(12, 6))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = torch.permute(targ_image, (1, 2, 0))

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title("Label: " + str(targ_label))

display_random_images(train_dataset, n=2, classes=None,seed=None)
display_random_images(train_dataset, n=2, classes=None,seed=None)

# Use offline ViT model

from transformers import ViTForImageClassification, ViTFeatureExtractor

num_classes = 2




# from transformers import DeiTForImageClassification

# model = DeiTForImageClassification.from_pretrained('/kaggle/working/deit-classification/deit_classification_model')

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "/kaggle/input/d/qjives/vitclassification/vit_classification_model"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class_weights = torch.tensor([0.1, 0.9])  # Example weights
class_weights = class_weights.to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

from tqdm import tqdm
num_epochs = 5

train_loss = []
val_acc = []
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.to(device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    train_loss.append(epoch_loss)
    # Validation step
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.to(device)
            valid_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_loader.dataset)
    accuracy = 100 * correct / total
    # val_loss.append(valid_loss)
    val_acc.append(accuracy)
    print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Test the model
model.eval()
test_loss = 0.0
correct = 0
total = 0

# plt.plot(train_loss, label='Training Loss')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

from sklearn.metrics import classification_report

# Test the model
model.eval()
test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store predictions and labels for classification report
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader.dataset)
accuracy = 100 * correct / total
print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Generate and print classification report
target_names = ['class_0', 'class_1']  # Replace with your actual class names
report = classification_report(all_labels, all_preds, target_names=target_names)
print(report)

model.eval()  # Set the model to evaluation mode

from torch.nn.functional import softmax

all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        
        # Get model outputs (logits)
        outputs = model(images).logits
        
        # Apply softmax to get probabilities
        probabilities = softmax(outputs, dim=1)
        
        # Get the probability of class 1 (assuming class 1 is at index 1)
        class_1_probs = probabilities[:, 1].cpu().numpy()
        
        all_probs.extend(class_1_probs)
# print(all_probs)

from sklearn.metrics import roc_curve, auc
import numpy as np

# Assuming `all_probs` contains the predicted probabilities for class 1
# and `all_labels` contains the true labels (0 or 1)
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

# Find the indices where TPR is above 0.8
tpr_threshold = 0.8
indices = np.where(tpr >= tpr_threshold)

# Filter the FPR and TPR to only include points where TPR >= 0.8
fpr_filtered = fpr[indices]
tpr_filtered = tpr[indices]

# Ensure the curve starts at FPR=0 for correct partial AUC calculation
fpr_filtered = np.concatenate([[0], fpr_filtered])
tpr_filtered = np.concatenate([[tpr_threshold], tpr_filtered])
partial_auc = auc(fpr_filtered, tpr_filtered)
print(f"Partial AUC with TPR >= 80%: {partial_auc:.4f}")

model.eval()  # Set the model to evaluation mode

from torch.nn.functional import softmax

all_probs = []

with torch.no_grad():
    for images in tqdm(test_loader_actual):
        images = images.to(device)
        
        # Get model outputs (logits)
        outputs = model(images).logits
        
        # Apply softmax to get probabilities
        probabilities = softmax(outputs, dim=1)
        
        # Get the probability of class 1 (assuming class 1 is at index 1)
        class_1_probs = probabilities[:, 1].cpu().numpy()
        
        all_probs.extend(class_1_probs)
print(all_probs)

df_to_submit = test_matadata['isic_id'].rename('isic_id')
df_to_submit.head()

probs_dataframe = pd.DataFrame(all_probs, columns=['target'], dtype = float)
probs_dataframe.head()
# df_to_submit['target']= probs_dataframe.astype(float)
# final_df = pd.concat(df_to_submit,probs_dataframe,0)
# final_df.head()

final_submission = pd.concat([df_to_submit,probs_dataframe],axis=1)
final_submission.head()
final_submission.to_csv('submission.csv', index = False)
