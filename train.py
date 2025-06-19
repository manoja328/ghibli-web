from PIL import Image
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class GhibliDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Ghibli', 'Non_Ghibli']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Supported image formats
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        
        # Load all images and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(self.supported_formats):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
                else:
                    print(f"Warning: Skipping unsupported file format: {img_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            # Try to open the image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default image (black) in case of error
            image = Image.new('RGB', (224, 224), color='black')
            
        label = self.labels[idx]
        
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error transforming image {img_path}: {str(e)}")
                # Return a default tensor in case of error
                image = torch.zeros((3, 224, 224))
            
        return image, label

class GhibliClassifier(nn.Module):
    def __init__(self):
        super(GhibliClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # Remove the last layer
        self.mobilenet.classifier = nn.Sequential()
        # Add our classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 1),  # MobileNetV2 output is 1280
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.mobilenet.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.view(-1)  # Ensure output is flattened

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device).view(-1)  # Ensure labels are properly shaped
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Remove .squeeze()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device).view(-1)  # Ensure labels are properly shaped
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Remove .squeeze()
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define data directory
    data_dir = "dataset"
    
    # Create dataset directory if it doesn't exist
    Path(data_dir).mkdir(exist_ok=True)
    
    # Create subdirectories for classes if they don't exist
    Path(os.path.join(data_dir, "Ghibli")).mkdir(exist_ok=True)
    Path(os.path.join(data_dir, "Non_Ghibli")).mkdir(exist_ok=True)
    
    print("Please place your images in the following directories:")
    print("\nSupported image formats: JPG, JPEG, PNG (case insensitive)")
    print(f"- {os.path.join(data_dir, 'Ghibli')} - for Ghibli-style images")
    print(f"- {os.path.join(data_dir, 'Non_Ghibli')} - for non-Ghibli images")
    
    # Check if there are images in the directories
    supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    ghibli_count = sum(len(list(Path(os.path.join(data_dir, "Ghibli")).glob(f"*{fmt}"))) for fmt in supported_formats)
    non_ghibli_count = sum(len(list(Path(os.path.join(data_dir, "Non_Ghibli")).glob(f"*{fmt}"))) for fmt in supported_formats)
    
    if ghibli_count == 0 or non_ghibli_count == 0:
        print("Error: Please add images to both Ghibli and Non_Ghibli directories")
        print("Supported formats:", ", ".join(supported_formats))
        return
    
    print(f"\nFound {ghibli_count} Ghibli images and {non_ghibli_count} non-Ghibli images")
    print("Supported image formats:", ", ".join(supported_formats))
    
    # Define data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = GhibliDataset(data_dir, transform=train_transform)
    val_dataset = GhibliDataset(data_dir, transform=val_transform)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    model = GhibliClassifier().to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    
    # Save the PyTorch model
    torch.save(model.state_dict(), 'model.pth')
    print("\nPyTorch model saved as 'model.pth'")
    
    # Export to ONNX
    model.eval()  # Set the model to evaluation mode
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Example input (1 image, 3 channels, 224x224)
    torch.onnx.export(
        model,                     # Model being exported
        dummy_input,              # Example input
        "model_pt.onnx",             # Output file
        export_params=True,       # Store the trained weights
        opset_version=11,         # ONNX version
        do_constant_folding=True, # Optimize constant folding
        input_names=['input'],    # Name of input node
        output_names=['output'],  # Name of output node
        dynamic_axes={            # Support dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("ONNX model saved as 'model.onnx'")

if __name__ == "__main__":
    main() 