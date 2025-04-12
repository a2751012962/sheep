import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Optional

class SampleOrganizer:
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.card_types = {
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
        
    def create_directory_structure(self):
        """创建数据集目录结构"""
        # 创建主目录
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每种卡牌类型创建子目录
        for card_type in self.card_types:
            (self.target_dir / card_type).mkdir(exist_ok=True)
            
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 调整大小
        image = cv2.resize(image, (50, 50))
        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv
        
    def classify_card(self, image: np.ndarray) -> Optional[str]:
        """根据颜色特征分类卡牌"""
        hsv = self.preprocess_image(image)
        best_match = None
        max_match_score = 0
        
        for card_type, info in self.card_types.items():
            match_score = 0
            for lower, upper in info['hsv_ranges']:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                match_ratio = np.sum(mask > 0) / mask.size
                match_score = max(match_score, match_ratio)
            
            if match_score > max_match_score and match_score > 0.3:
                max_match_score = match_score
                best_match = card_type
                
        return best_match
        
    def organize_samples(self):
        """组织样本数据"""
        print("开始组织样本数据...")
        self.create_directory_structure()
        
        # 支持的图片格式
        supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        # 处理源目录中的所有图片
        for image_path in self.source_dir.glob('**/*'):
            if image_path.suffix.lower() in supported_extensions:
                try:
                    # 读取图片
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"无法读取图片: {image_path}")
                        continue
                        
                    # 分类卡牌
                    card_type = self.classify_card(image)
                    if card_type:
                        # 创建目标路径
                        target_path = self.target_dir / card_type / image_path.name
                        # 复制文件
                        shutil.copy2(image_path, target_path)
                        print(f"已分类: {image_path.name} -> {card_type}")
                    else:
                        print(f"无法分类: {image_path.name}")
                        
                except Exception as e:
                    print(f"处理图片时出错 {image_path}: {str(e)}")
                    
        print("样本数据组织完成！")
        
    def get_statistics(self) -> Dict[str, int]:
        """获取样本统计信息"""
        stats = {}
        for card_type in self.card_types:
            type_dir = self.target_dir / card_type
            if type_dir.exists():
                count = len(list(type_dir.glob('*')))
                stats[card_type] = count
        return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description='组织卡牌样本数据')
    parser.add_argument('--source', type=str, required=True, help='源图片目录')
    parser.add_argument('--target', type=str, required=True, help='目标数据集目录')
    args = parser.parse_args()
    
    organizer = SampleOrganizer(args.source, args.target)
    organizer.organize_samples()
    
    # 打印统计信息
    stats = organizer.get_statistics()
    print("\n样本统计:")
    for card_type, count in stats.items():
        print(f"{card_type}: {count} 个样本")

if __name__ == '__main__':
    main() 