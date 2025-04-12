<<<<<<< HEAD
# 羊了个羊自动解题器

这是一个用于自动解决羊了个羊游戏的工具。它使用计算机视觉和深度学习技术来识别游戏中的卡牌，并使用决策算法来找到最优解。

## 项目结构

```
sheep/
├── data/                  # 数据存储
│   ├── dataset/          # 处理后的数据集
│   │   ├── train/       # 训练集样本
│   │   ├── val/         # 验证集样本
│   │   └── test/        # 测试集样本
│   └── temp/            # 临时文件
├── docs/                 # 项目文档
├── images/              # 图片资源
│   ├── screenshots/     # 游戏截图
│   └── templates/       # 卡片模板
├── image_recognition/   # 图像识别模块
│   ├── card_detector.py # 卡片检测器
│   ├── cnn_model.py     # CNN模型定义
│   └── game_state.py    # 游戏状态识别
├── decision_making/     # 决策模块
│   ├── strategy.py      # 游戏策略
│   └── rl_agent.py      # 强化学习代理
├── learning/            # 学习模块
│   └── experience_replay.py # 经验回放
├── models/              # 训练好的模型
├── tests/               # 测试文件
│   ├── test_image.py    # 图像处理测试
│   └── test_window.py   # 窗口操作测试
├── tools/               # 工具脚本
│   ├── utils/          # 工具函数
│   │   ├── prepare_dataset.py    # 数据集准备
│   │   ├── sample_organizer.py   # 样本组织
│   │   └── template_labeler.py   # 模板标注
│   ├── template_extractor.py     # 模板提取
│   └── train_model.py           # 模型训练
├── main.py              # 主程序入口
├── README.md            # 项目说明
└── requirements.txt     # 依赖包列表
```

## 需要清理的文件

以下文件功能重复或已过时，需要整理或删除：

1. tools/ 目录下的重复文件：
   - process_screenshots.py (合并到 card_detector.py)
   - process_cards.py (合并到 card_detector.py)
   - save_cards.py (可删除)
   - train_classifier.py (合并到 train_model.py)
   - process_all.py (可删除)
   - template_labeler.py (移动到 tools/utils/)
   - card_classifier.py (合并到 cnn_model.py)
   - simple_cropper.py (合并到 card_detector.py)
   - test_image.py (移动到 tests/)
   - image_cropper.py (合并到 card_detector.py)
   - test_window.py (移动到 tests/)

2. 需要创建的新目录：
   - tests/ : 存放测试文件
   - tools/utils/ : 存放辅助工具
   - docs/ : 存放文档

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/sheep-solver.git
cd sheep-solver
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 提取卡牌模板：
```bash
python tools/template_extractor.py images/screenshots/game.png images/templates
```

2. 训练模型：
```bash
python tools/train_model.py
```

3. 运行解题器：
```bash
python main.py
```

## 依赖项

- Python 3.8+
- OpenCV
- PyTorch
- NumPy
- Pillow

## 贡献

欢迎提交 Pull Request 或创建 Issue。

## 许可证

MIT License 
=======
# sheep
>>>>>>> d1b411e347c1dbf5d2d30dbf0828bd283efb0dec
