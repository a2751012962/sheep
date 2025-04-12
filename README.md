# 羊了个羊自动解题器

这是一个用于自动解决羊了个羊游戏的工具。它使用计算机视觉和深度学习技术来识别游戏中的卡牌，并使用决策算法来找到最优解。

## 功能特点

- 🎯 自动识别游戏卡牌
- 🤖 基于深度学习的卡牌分类
- 🎮 智能游戏策略生成
- 📊 实时游戏状态分析
- 🔄 自动化游戏操作

## 项目结构

```
sheep/
├── data/                  # 数据存储
│   └── dataset/          # 处理后的数据集
│       ├── train/        # 训练数据（80%）
│       ├── val/          # 验证数据（10%）
│       └── test/         # 测试数据（10%）
├── docs/                 # 项目文档
├── images/              # 图片资源
│   ├── screenshots/     # 游戏原始截图
│   ├── cards/          # 单个卡片原始图片
│   └── templates/      # 处理后的标准模板
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

## 数据组织说明

### 图片资源 (`images/`)
- `screenshots/`: 存放游戏截图，用于提取卡片模板
- `cards/`: 存放单个卡片的原始图片，作为基准样本
- `templates/`: 存放经过处理的标准模板，按类别分类（如 red_panda, toucan 等）

### 数据集 (`data/dataset/`)
- `train/`: 训练集，包含约80%的数据样本
- `val/`: 验证集，包含约10%的数据样本
- `test/`: 测试集，包含约10%的数据样本

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/a2751012962/sheep.git
cd sheep
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：
   ```bash
   # 将游戏截图放入 images/screenshots/
   # 将单个卡片图片放入 images/cards/
   ```

2. 提取模板：
   ```bash
   python tools/template_extractor.py images/cards/* images/templates
   ```

3. 准备数据集：
   ```bash
   python tools/utils/prepare_dataset.py
   ```

4. 训练模型：
   ```bash
   python tools/train_model.py
   ```

5. 运行解题器：
   ```bash
   python main.py
   ```

## 技术栈

- 🐍 Python 3.8+
- 🖼️ OpenCV - 图像处理和卡牌检测
- 🧠 PyTorch - 深度学习模型训练
- 📊 NumPy - 数值计算和数据处理
- 🎨 Pillow - 图像处理和增强

## 开发指南

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 贡献

欢迎提交 Pull Request 或创建 Issue。任何形式的贡献都将被感激。

## 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件了解更多细节 
