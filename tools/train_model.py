import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
from template_extractor import CARD_TYPES, create_model

class CardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CARD_TYPES.keys())}
        
        # 收集所有模板
        for card_type in CARD_TYPES.keys():
            type_dir = self.data_dir / card_type
            if type_dir.exists():
                for img_path in type_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[card_type]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(data_dir, model_path, num_epochs=50):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = CardDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print("错误：没有找到训练数据")
        return
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(CARD_TYPES))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print(f"开始训练，使用设备: {device}")
    print(f"数据集大小: {len(dataset)} 个样本")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.2f}%')
    
    print("训练完成")
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='训练卡牌分类模型')
    parser.add_argument('--data_dir', default='images/templates', help='模板目录路径')
    parser.add_argument('--model_path', default='models/card_classifier.pth', help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    args = parser.parse_args()
    
    train_model(args.data_dir, args.model_path, args.epochs)

if __name__ == '__main__':
    main() 