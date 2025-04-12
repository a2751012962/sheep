import os
import shutil
from pathlib import Path

def prepare_dataset(template_dir, dataset_dir):
    """
    将模板复制到数据集目录
    :param template_dir: 模板目录路径
    :param dataset_dir: 数据集目录路径
    """
    template_dir = Path(template_dir)
    dataset_dir = Path(dataset_dir)
    
    # 确保数据集目录存在
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历模板目录
    for category_dir in template_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        # 获取类别名称
        category = category_dir.name
        
        # 创建对应的数据集目录
        target_dir = dataset_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制模板文件
        for template_file in category_dir.iterdir():
            if template_file.is_file():
                shutil.copy2(template_file, target_dir / template_file.name)
                print(f"Copied {template_file} to {target_dir}")

if __name__ == "__main__":
    # 设置路径
    template_dir = Path("images/templates")
    train_dir = Path("dataset/train")
    val_dir = Path("dataset/val")
    test_dir = Path("dataset/test")
    
    # 准备训练集
    print("Preparing training set...")
    prepare_dataset(template_dir, train_dir)
    
    # 准备验证集（从训练集中随机选择20%的样本）
    print("Preparing validation set...")
    prepare_dataset(template_dir, val_dir)
    
    # 准备测试集（从训练集中随机选择10%的样本）
    print("Preparing test set...")
    prepare_dataset(template_dir, test_dir) 