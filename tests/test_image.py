import cv2
import numpy as np
from pathlib import Path
import sys
import os

def main():
    # 获取当前工作目录的绝对路径
    workspace_dir = Path(os.getcwd())
    print(f"当前工作目录: {workspace_dir}", flush=True)
    
    # 构建图片的绝对路径
    image_path = workspace_dir / 'images' / 'screenshots' / 'game_001.png'
    print(f"图片绝对路径: {image_path}", flush=True)
    
    # 检查文件是否存在
    if not image_path.exists():
        print(f"错误: 文件不存在: {image_path}", flush=True)
        return
    
    print(f"文件大小: {image_path.stat().st_size} 字节", flush=True)
    
    # 尝试读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误: 无法读取图片: {image_path}", flush=True)
        return
    
    print(f"图片尺寸: {image.shape}", flush=True)
    print("图片读取成功!", flush=True)
    
    # 保存一个测试图片
    test_output = workspace_dir / 'images' / 'screenshots' / 'test_output.png'
    cv2.imwrite(str(test_output), image)
    print(f"已保存测试图片: {test_output}", flush=True)

if __name__ == '__main__':
    main() 