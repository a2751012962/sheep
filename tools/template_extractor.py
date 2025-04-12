import cv2
import numpy as np
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import shutil

def create_model(num_classes):
    """创建基于ResNet18的迁移学习模型"""
    model = models.resnet18(pretrained=True)
    
    # 冻结大部分层
    for param in model.parameters():
        param.requires_grad = False
        
    # 替换最后的全连接层
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

# 定义所有卡牌类型及其特征
CARD_TYPES = {
    'toucan': {'name': '大嘴鸟', 'hsv_ranges': [([0, 50, 50], [10, 255, 255])]},
    'donut': {'name': '甜甜圈', 'hsv_ranges': [([140, 50, 50], [170, 255, 255])]},
    'chameleon': {'name': '变色龙', 'hsv_ranges': [([35, 50, 50], [85, 255, 255])]},
    'red_panda': {'name': '小熊猫', 'hsv_ranges': [([0, 50, 50], [20, 255, 255])]},
    'panda': {'name': '大熊猫', 'hsv_ranges': [([0, 0, 0], [180, 30, 255])]},
    'flamingo': {'name': '火烈鸟', 'hsv_ranges': [([150, 50, 50], [180, 255, 255])]},
    'koala': {'name': '考拉', 'hsv_ranges': [([0, 0, 100], [180, 30, 200])]},
    'frog': {'name': '青蛙', 'hsv_ranges': [([35, 100, 100], [85, 255, 255])]},
    'sloth': {'name': '树懒', 'hsv_ranges': [([20, 50, 100], [30, 150, 200])]},
    'blue_butterfly': {'name': '蓝蝴蝶', 'hsv_ranges': [([100, 150, 150], [140, 255, 255])]},
    'green_butterfly': {'name': '绿蝴蝶', 'hsv_ranges': [([35, 100, 100], [85, 255, 255])]},
    'tree': {'name': '树木', 'hsv_ranges': [([35, 100, 100], [85, 255, 255])]},
    'snail': {'name': '蜗牛', 'hsv_ranges': [([20, 100, 100], [30, 255, 255])]},
    'flower': {'name': '花朵', 'hsv_ranges': [([150, 100, 150], [180, 255, 255])]},
    'black_bear': {'name': '黑熊', 'hsv_ranges': [([0, 0, 0], [180, 255, 50])]},
    'monkey': {'name': '猴子', 'hsv_ranges': [([20, 50, 100], [30, 150, 200])]},
    'elephant': {'name': '大象', 'hsv_ranges': [([100, 100, 150], [140, 255, 255])]},
    'palm_tree': {'name': '棕榈树', 'hsv_ranges': [([35, 100, 100], [85, 255, 255])]}
}

# 图像预处理转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path='models/card_classifier.pth'):
    """加载预训练的CNN模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(CARD_TYPES))
    
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.to(device)
    else:
        print(f"警告: 未找到预训练模型 {model_path}，将仅使用颜色特征进行识别")
        return None

def get_color_histogram(image):
    """计算颜色直方图特征"""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def get_shape_features(contour):
    """计算形状特征"""
    # 计算周长和面积
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    # 计算圆度
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 计算矩形度
    x, y, w, h = cv2.boundingRect(contour)
    rectangularity = (w * h) / area if area > 0 else 0
    
    # 计算长宽比
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # 计算紧凑度
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    return circularity, rectangularity, aspect_ratio, solidity

def is_similar_template(template1, template2, threshold=0.85):
    """检查两个模板是否相似"""
    # 颜色直方图比较
    hist1 = get_color_histogram(template1)
    hist2 = get_color_histogram(template2)
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 结构相似度
    ssim = cv2.matchTemplate(template1, template2, cv2.TM_CCOEFF_NORMED)[0][0]
    
    return hist_similarity > threshold and ssim > threshold

def extract_templates(image_path, output_dir):
    """提取卡牌模板"""
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 加载CNN模型
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model = model.to(device)
        print("已加载CNN模型")
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 用于标记检测到的卡牌
    marked_image = image.copy()
    
    # 清空现有的模板目录
    for card_type in CARD_TYPES:
        type_dir = output_dir / card_type
        type_dir.mkdir(exist_ok=True)
        # 清空目录中的文件
        for file in type_dir.glob('*.png'):
            try:
                file.unlink()
            except PermissionError:
                print(f"警告：无法删除文件 {file}")
    
    # 创建全局掩码用于初步检测卡牌
    global_mask = np.zeros_like(hsv[:,:,0])
    for info in CARD_TYPES.values():
        for lower, upper in info['hsv_ranges']:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            global_mask = cv2.bitwise_or(global_mask, mask)
    
    # 形态学操作改进掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, kernel)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, kernel)
    
    # 边缘检测
    edges = cv2.Canny(global_mask, 50, 150)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area < 150:  # 降低面积阈值
            continue
        
        # 获取形状特征
        circularity, rectangularity, aspect_ratio, solidity = get_shape_features(contour)
        
        # 根据形状特征过滤
        if circularity < 0.3 or rectangularity < 0.5 or solidity < 0.7:
            continue
        
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤不合适的尺寸
        if w < 15 or h < 15 or w > 300 or h > 300:
            continue
        
        # 提取卡牌区域
        card = image[y:y+h, x:x+w]
        
        # 使用CNN进行分类
        if model is not None:
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
            # 应用预处理
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                if confidence.item() > 0.7:  # 置信度阈值
                    card_type = list(CARD_TYPES.keys())[predicted.item()]
                    type_dir = output_dir / card_type
                    
                    # 调整大小
                    resized_card = cv2.resize(card, (50, 50))
                    
                    # 检查是否与现有模板重复
                    is_duplicate = False
                    for template_path in type_dir.glob('*.png'):
                        template = cv2.imread(str(template_path))
                        if is_similar_template(resized_card, template):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        # 保存模板
                        template_path = type_dir / f'template_{len(list(type_dir.glob("*.png")))}.png'
                        cv2.imwrite(str(template_path), resized_card)
                        
                        # 在原图上标记
                        cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(marked_image, CARD_TYPES[card_type]['name'], (x, y-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        print(f"发现 {CARD_TYPES[card_type]['name']} (置信度: {confidence.item():.2f})")
        else:
            # 如果没有CNN模型，使用颜色特征进行分类
            best_match = None
            max_match_score = 0
            
            for card_type, info in CARD_TYPES.items():
                match_score = 0
                card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
                
                for lower, upper in info['hsv_ranges']:
                    mask = cv2.inRange(card_hsv, np.array(lower), np.array(upper))
                    match_ratio = np.sum(mask > 0) / mask.size
                    match_score = max(match_score, match_ratio)
                
                if match_score > max_match_score and match_score > 0.3:
                    max_match_score = match_score
                    best_match = card_type
            
            if best_match:
                type_dir = output_dir / best_match
                resized_card = cv2.resize(card, (50, 50))
                
                # 检查是否与现有模板重复
                is_duplicate = False
                for template_path in type_dir.glob('*.png'):
                    template = cv2.imread(str(template_path))
                    if is_similar_template(resized_card, template):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # 保存模板
                    template_path = type_dir / f'template_{len(list(type_dir.glob("*.png")))}.png'
                    cv2.imwrite(str(template_path), resized_card)
                    
                    # 在原图上标记
                    cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(marked_image, CARD_TYPES[best_match]['name'], (x, y-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 保存标记后的图像
    cv2.imwrite(str(output_dir / 'marked_cards.png'), marked_image)

def main():
    parser = argparse.ArgumentParser(description='从游戏截图中提取卡牌模板')
    parser.add_argument('image_path', help='输入图像路径')
    parser.add_argument('output_dir', help='输出目录路径')
    args = parser.parse_args()
    
    extract_templates(args.image_path, args.output_dir)

if __name__ == '__main__':
    main() 