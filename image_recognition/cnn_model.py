<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import json

class CardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.png'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CardClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CardClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # 预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        return self.resnet(x)
    
    def preprocess_image(self, image):
        """图像预处理"""
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image)
        return self.transform(pil_image).unsqueeze(0)
    
    def predict(self, image, device=None):
        """预测单张图片的类别"""
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            input_tensor = self.preprocess_image(image).to(device)
            output = self(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item()

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {accuracy:.2f}%')

def create_model(num_classes, weights_path=None):
    """创建模型实例，可选择性加载预训练权重"""
    model = CardClassifier(num_classes)
    
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path}")
    
    return model

def save_model(model, save_dir, class_mapping=None):
    """保存模型和类别映射"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型权重
    weights_path = save_dir / 'card_classifier.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")
    
    # 保存类别映射
    if class_mapping:
        mapping_path = save_dir / 'class_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f)
=======
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import json

class CardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.png'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CardClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CardClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # 预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        return self.resnet(x)
    
    def preprocess_image(self, image):
        """图像预处理"""
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image)
        return self.transform(pil_image).unsqueeze(0)
    
    def predict(self, image, device=None):
        """预测单张图片的类别"""
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            input_tensor = self.preprocess_image(image).to(device)
            output = self(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item()

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {accuracy:.2f}%')

def create_model(num_classes, weights_path=None):
    """创建模型实例，可选择性加载预训练权重"""
    model = CardClassifier(num_classes)
    
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path}")
    
    return model

def save_model(model, save_dir, class_mapping=None):
    """保存模型和类别映射"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型权重
    weights_path = save_dir / 'card_classifier.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")
    
    # 保存类别映射
    if class_mapping:
        mapping_path = save_dir / 'class_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f)
>>>>>>> d1b411e347c1dbf5d2d30dbf0828bd283efb0dec
        print(f"Class mapping saved to {mapping_path}") 