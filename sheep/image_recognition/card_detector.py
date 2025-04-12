import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import shutil
import os
import stat
from .game_state import Card, GameState
from .cnn_model import CardClassifier

class CardDetector:
    """卡牌检测器类"""
    
    # 定义所有卡牌类型及其特征颜色范围（HSV空间）
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
    
    def __init__(self, template_dir: str = 'images/templates', model_path: str = None):
        self.template_dir = Path(template_dir)
        self.model = None
        if model_path:
            self.model = CardClassifier(len(self.CARD_TYPES))
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        
        # 卡牌检测参数
        self.min_card_size = 30  # 最小卡牌尺寸
        self.max_card_size = 100  # 最大卡牌尺寸
        self.match_threshold = 0.8  # 模板匹配阈值
        
    @staticmethod
    def remove_readonly(func, path, _):
        """清除只读属性并重试删除"""
        os.chmod(path, stat.S_IWRITE)
        func(path)
    
    def process_single_card(self, image_path: Path, output_dir: Path):
        """处理单张卡牌图片"""
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        
        # 创建一个白色背景的大图像，将卡牌放在中间
        bg_size = max(image.shape[0], image.shape[1]) * 2
        background = np.ones((bg_size, bg_size, 3), dtype=np.uint8) * 255
        
        # 计算居中位置
        y_offset = (bg_size - image.shape[0]) // 2
        x_offset = (bg_size - image.shape[1]) // 2
        
        # 将卡牌图像放在背景中间
        background[y_offset:y_offset+image.shape[0], 
                  x_offset:x_offset+image.shape[1]] = image
        
        # 检测卡牌
        cards = self.detect_cards(background)
        
        # 保存检测到的卡牌
        for i, card in enumerate(cards):
            if card.type:
                type_dir = output_dir / card.type
                type_dir.mkdir(exist_ok=True)
                x, y = card.position
                w = h = self.max_card_size
                card_img = background[y-h//2:y+h//2, x-w//2:x+w//2]
                if card_img.size > 0:
                    cv2.imwrite(str(type_dir / f"{image_path.stem}_{i}.png"), card_img)
    
    def process_screenshots(self, screenshots_dir: Path, output_dir: Path):
        """处理所有截图并提取卡牌"""
        # 清空之前的模板
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir, onerror=self.remove_readonly)
            except Exception as e:
                print(f"警告：无法完全清除旧的模板目录: {e}")
                # 尝试删除所有可以删除的文件
                for item in output_dir.glob('**/*'):
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, onerror=self.remove_readonly)
                    except:
                        pass
        
        # 创建输出目录和所有卡牌类型的子目录
        output_dir.mkdir(parents=True, exist_ok=True)
        for card_type in self.CARD_TYPES:
            (output_dir / card_type).mkdir(exist_ok=True)
        
        # 处理每个截图
        total_processed = 0
        supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        print("\n开始处理截图...")
        for image_path in screenshots_dir.glob('**/*'):
            if image_path.suffix.lower() in supported_extensions:
                print(f"\n处理截图: {image_path.name}")
                self.process_single_card(image_path, output_dir)
                total_processed += 1
        
        if total_processed == 0:
            print(f"\n在 {screenshots_dir} 目录下没有找到图片文件")
        else:
            print(f"\n处理完成！共处理了 {total_processed} 个截图")
            print(f"提取的卡牌模板已保存到: {output_dir}")
            
            # 统计每种类型的卡牌数量
            print("\n各类型卡牌统计：")
            for card_type in self.CARD_TYPES:
                type_dir = output_dir / card_type
                if type_dir.exists():
                    count = len(list(type_dir.glob('*.png')))
                    if count > 0:
                        print(f"{self.CARD_TYPES[card_type]['name']}: {count} 个")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预处理图像"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建全局掩码
        global_mask = np.zeros_like(hsv[:,:,0])
        for info in self.CARD_TYPES.values():
            for lower, upper in info['hsv_ranges']:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                global_mask = cv2.bitwise_or(global_mask, mask)
        
        # 应用形态学操作来清理噪声
        kernel = np.ones((3,3), np.uint8)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, kernel)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, kernel)
        
        return hsv, global_mask
    
    def detect_cards(self, image: np.ndarray) -> List[Card]:
        """检测图像中的所有卡牌"""
        cards = []
        hsv, mask = self.preprocess_image(image)
        
        # 找到所有可能的卡牌轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            # 计算轮廓的基本属性
            area = cv2.contourArea(contour)
            if area < self.min_card_size * self.min_card_size or \
               area > self.max_card_size * self.max_card_size:
                continue
            
            # 获取卡牌区域
            x, y, w, h = cv2.boundingRect(contour)
            card_roi = image[y:y+h, x:x+w]
            
            # 识别卡牌类型
            card_type = self._recognize_card_type(card_roi)
            if card_type:
                # 判断卡牌是否可见
                is_visible = self._check_card_visibility(card_roi)
                
                # 创建卡牌对象
                card = Card(
                    type=card_type,
                    position=(x + w//2, y + h//2),  # 使用卡牌中心点作为位置
                    is_visible=is_visible
                )
                cards.append(card)
        
        return cards
    
    def _recognize_card_type(self, card_image: np.ndarray) -> Optional[str]:
        """识别卡牌类型"""
        if self.model is not None:
            # 使用CNN模型识别
            predicted, confidence = self.model.predict(card_image)
            if confidence > 0.7:  # 置信度阈值
                return list(self.CARD_TYPES.keys())[predicted]
        
        # 如果没有模型或置信度不够，使用颜色特征识别
        hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
        best_match = None
        max_match_score = 0
        
        for card_type, info in self.CARD_TYPES.items():
            match_score = 0
            for lower, upper in info['hsv_ranges']:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                match_ratio = np.sum(mask > 0) / mask.size
                match_score = max(match_score, match_ratio)
            
            if match_score > max_match_score and match_score > 0.3:
                max_match_score = match_score
                best_match = card_type
        
        return best_match
    
    def _check_card_visibility(self, card_image: np.ndarray) -> bool:
        """检查卡牌是否可见（未被遮挡）"""
        # 转换为灰度图
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        # 计算平均亮度和标准差
        mean, std = cv2.meanStdDev(gray)
        # 如果平均亮度较高且标准差较大（表示有清晰的图案），认为卡牌可见
        return mean[0][0] > 128 and std[0][0] > 30
        
    def detect_game_state(self, image: np.ndarray) -> GameState:
        """检测完整的游戏状态"""
        game_state = GameState()
        
        # 检测所有卡牌
        cards = self.detect_cards(image)
        for card in cards:
            game_state.add_card(card)
            
        # 检测底部槽状态
        self._detect_slot_state(image, game_state)
        
        # 检测道具和能量状态
        self._detect_power_and_items(image, game_state)
        
        return game_state
        
    def _detect_slot_state(self, image: np.ndarray, game_state: GameState):
        """检测底部槽的状态"""
        # 提取底部区域
        height, width = image.shape[:2]
        slot_region = image[int(height*0.8):, :]
        
        # 检测槽中的卡牌
        slot_cards = self.detect_cards(slot_region)
        for card in slot_cards:
            game_state.add_to_slot(card)
            
    def _detect_power_and_items(self, image: np.ndarray, game_state: GameState):
        """检测能量和道具状态"""
        # 提取右上角区域
        height, width = image.shape[:2]
        power_region = image[:int(height*0.2), -int(width*0.2):]
        
        # 检测闪电能量图标
        power_count = self._detect_power_icons(power_region)
        game_state.power = power_count
        
        # 检测道具图标
        game_state.shuffle_count = self._detect_item_count(image, 'shuffle')
        game_state.remove_count = self._detect_item_count(image, 'remove')
        game_state.undo_count = self._detect_item_count(image, 'undo')
        
    def _detect_power_icons(self, power_region: np.ndarray) -> int:
        """检测闪电能量图标数量"""
        # 转换为HSV
        hsv = cv2.cvtColor(power_region, cv2.COLOR_BGR2HSV)
        
        # 黄色闪电的HSV范围
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        return len(contours)
        
    def _detect_item_count(self, image: np.ndarray, item_type: str) -> int:
        """检测特定道具的数量"""
        # 这里需要根据实际游戏界面实现具体的检测逻辑
        # 简化版本，默认返回1
        return 1 