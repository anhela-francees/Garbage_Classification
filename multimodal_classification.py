import torch
import glob
import matplotlib.pylab as plt
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import resnet18
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
import os
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau


TRAIN_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"



######################################## Data Preprocessing ########################################

class ProcessData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.texts = []
        self.labels = []
        self.class_folders = sorted(os.listdir(root_dir))  # Assuming class folders are sorted
        self.label_map = {class_name: idx for idx, class_name in enumerate(self.class_folders)} # Map each class name to a numeric label

        for class_name in self.class_folders:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                file_names = os.listdir(class_path)
                for file_name in file_names:
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        file_name_no_ext, _ = os.path.splitext(file_name)
                        text = file_name_no_ext.replace('_', ' ')
                        text_without_digits = re.sub(r'\d+', '', text)
                        self.image_paths.append(file_path) # Append image paths
                        self.texts.append(text_without_digits) # Append preprocessed texts
                        self.labels.append(self.label_map[class_name]) # Append preprocessed labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, text, label
        

# Training and validation transforms
torchvision_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize the images to 224x224 pixels
    transforms.RandomHorizontalFlip(), # Apply a random horizontal flip to the images
    transforms.RandomVerticalFlip(), # Apply a random vertical flip to the images
    transforms.RandomRotation(15), # Apply a random rotaton of 15 degrees to the images
    transforms.ToTensor(), # Convert the images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the tensors
])

# Testing transforms
torchvision_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Inputting data into ProcessData class (loading the datasets)
train_dataset = ProcessData(root_dir=TRAIN_PATH, transform=torchvision_transform)
val_dataset = ProcessData(root_dir=VAL_PATH, transform=torchvision_transform)
test_dataset = ProcessData(root_dir=TEST_PATH, transform=torchvision_transform_test)

# Define batch size and number of workers
batch_size = 32
num_workers = 4

# Creating dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


######################################## Text Embedding ########################################
class TextModel(nn.Module):
    def __init__(self, num_classes, freeze_layers=True):
        super(TextModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        if freeze_layers:
           for name, param in self.distilbert.named_parameters():
              if "layer" in name and int(name.split(".")[2]) < 4:  # Freeze first few layers
                  param.requires_grad = False
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return output

######################################## Image Embedding ########################################
class ImageModel(nn.Module):
    def __init__(self, num_classes, input_shape, transfer=True):
        super(ImageModel, self).__init__()
        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Load ResNet-34 and remove the final fully connected layer
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])  # Exclude FC layer

    def forward(self, x):
        x = self.feature_extractor(x)  # Shape should be (batch_size, 512, 1, 1) for resnet34
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
        #print(f"ImageModel Output Shape (after flattening): {x.shape}")  # Debug print statement
        return x


######################################## Fused Models ########################################
class GarbageClassification(nn.Module):
    def __init__(self, num_classes):
        super(GarbageClassification, self).__init__()
        self.text_encoder = TextModel(num_classes)
        self.image_encoder = ImageModel(num_classes, input_shape=(3,224,224), transfer=True)
        
        # The fusion layer should match the concatenated size (768 + 512 = 1280)
        self.fusion_layer = nn.Linear(1280, 512)
        self.fusion_dropout = nn.Dropout(0.5)  # Adding dropout after fusion layer

        # Final classifier with an additional dropout layer for regularization
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout in the final classifier
            nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Extract image and text features
        image_features = self.image_encoder(images)  # Shape: (batch_size, 512)
        #print(f"Image Features Shape: {image_features.shape}")

        text_features = self.text_encoder(input_ids, attention_mask)  # Shape: (batch_size, 768)
        #print(f"Text Features Shape: {text_features.shape}")

        # Concatenate the image and text features
        fused_features = torch.cat((image_features, text_features), dim=1)  # Shape: (batch_size, 1280)
        #print(f"Fused Features Shape (after concatenation): {fused_features.shape}")

        # Apply fusion layer followed by dropout
        fused_features = self.fusion_layer(fused_features)  # Shape: (batch_size, 512)
        fused_features = self.fusion_dropout(fused_features)  # Apply dropout
        #print(f"Fused Features Shape (after fusion layer and dropout): {fused_features.shape}")

        # Final logits from the classifier
        logits = self.classifier(fused_features)  # Shape: (batch_size, num_classes)
        #print(f"Logits Shape: {logits.shape}")

        return logits
######################################## Training ########################################

num_classes = 4
nepochs = 10

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = GarbageClassification(num_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
criterion = nn.CrossEntropyLoss()

PATH_BEST = './garbage_net.pth' # Path to save the best model

def tokenize_batch(texts):
    # Use the tokenizer to encode the text inputs (padding and truncating automatically handled)
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Define a function to train the model for one epoch
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, text_inputs, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        text_inputs = tokenize_batch(text_inputs)
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    return running_loss / len(data_loader), correct / total

# Define a function to evaluate the model on the validation set
def eval_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during validation
        for images, text_inputs, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # Tokenize the text inputs
            text_inputs = tokenize_batch(text_inputs)  # Ensure text_inputs is tokenized

            # Now access the tokenized data correctly
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)

            # Forward pass
            outputs = model(images, input_ids, attention_mask)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Track accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

# Training loop
best_val_accuracy = 0.0
patience = 5
early_stop_count = 0

for epoch in range(nepochs):  # Increased epochs
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)

    # Step the scheduler with validation accuracy (or val_loss if you prefer)
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), PATH_BEST)
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break

print("Training complete!")
print(f"Best model saved at {PATH_BEST}")

model.load_state_dict(torch.load(PATH_BEST))
test_loss, test_accuracy = eval_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


