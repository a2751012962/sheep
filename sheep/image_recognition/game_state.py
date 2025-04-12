from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

@dataclass
class Card:
    """卡牌类"""
    type: str  # 卡牌类型：树木、树懒、大象等
    position: Tuple[int, int]  # 卡牌在界面中的位置
    is_visible: bool  # 是否可见（未被遮挡）
    is_selected: bool = False  # 是否被选中
    
class CardStats:
    """卡牌统计类"""
    TOTAL_CARDS = {
        'toucan': 15,     # 大嘴鸟
        'koala': 15,      # 考拉
        'donut': 15,      # 甜甜圈
        'chameleon': 15,  # 变色龙
        'frog': 15,       # 青蛙
        'sloth': 15,      # 树懒
        'blue_butterfly': 15,   # 蓝蝴蝶
        'green_butterfly': 15,  # 绿蝴蝶
        'tree': 15,       # 树木
        'red_panda': 15,  # 小熊猫
        'panda': 15,      # 大熊猫
        'snail': 15,      # 蜗牛
        'flamingo': 15,   # 火烈鸟
        'flower': 15,     # 花朵
        'black_bear': 15, # 黑熊
        'monkey': 15,     # 猴子
        'elephant': 15,   # 大象
        'palm_tree': 15,  # 棕榈树
    }
    
    def __init__(self):
        # 初始化每种卡牌的剩余数量
        self.remaining_cards = self.TOTAL_CARDS.copy()
        # 已消除的卡牌（每种卡牌消除数量必须是3的倍数）
        self.eliminated_cards = {card_type: 0 for card_type in self.TOTAL_CARDS}
        # 场上可见的卡牌
        self.visible_cards = {card_type: 0 for card_type in self.TOTAL_CARDS}
        # 场上被遮挡的卡牌
        self.hidden_cards = {card_type: 0 for card_type in self.TOTAL_CARDS}
        # 槽中的卡牌
        self.slot_cards = {card_type: 0 for card_type in self.TOTAL_CARDS}
        
    def update_card_count(self, card_type: str, status: str, count: int = 1):
        """更新卡牌计数"""
        if card_type not in self.TOTAL_CARDS:
            return
            
        if status == 'eliminated':
            self.eliminated_cards[card_type] += count
        elif status == 'visible':
            self.visible_cards[card_type] += count
        elif status == 'hidden':
            self.hidden_cards[card_type] += count
        elif status == 'slot':
            self.slot_cards[card_type] += count
            
        # 更新剩余卡牌数量
        self.remaining_cards[card_type] = (
            self.TOTAL_CARDS[card_type] -
            self.eliminated_cards[card_type] -
            self.visible_cards[card_type] -
            self.hidden_cards[card_type] -
            self.slot_cards[card_type]
        )
        
    def get_total_remaining(self) -> int:
        """获取总的剩余卡牌数量"""
        return sum(self.remaining_cards.values())
    
    def get_card_probability(self, card_type: str) -> float:
        """计算下一张牌是特定类型的概率"""
        total_remaining = self.get_total_remaining()
        if total_remaining == 0:
            return 0.0
        return self.remaining_cards[card_type] / total_remaining
    
    def get_matching_probability(self, card_type: str) -> float:
        """计算找到配对的概率"""
        visible_count = self.visible_cards[card_type]
        remaining_count = self.remaining_cards[card_type]
        
        if visible_count >= 2:  # 已经有可配对的牌
            return 1.0
        elif visible_count == 1:  # 需要再找一张
            return remaining_count / self.get_total_remaining() if self.get_total_remaining() > 0 else 0.0
        else:  # 需要找两张
            if self.get_total_remaining() < 2:
                return 0.0
            # 超几何分布概率计算
            p1 = remaining_count / self.get_total_remaining()
            p2 = (remaining_count - 1) / (self.get_total_remaining() - 1)
            return p1 * p2
    
    def get_eliminated_groups(self, card_type: str) -> int:
        """获取已消除的组数（每组3张）"""
        return self.eliminated_cards[card_type] // 3
        
    def get_remaining_groups(self, card_type: str) -> int:
        """获取剩余可消除的组数"""
        return self.remaining_cards[card_type] // 3
        
    def get_total_groups(self) -> int:
        """获取所有卡牌的总组数"""
        return sum(self.get_remaining_groups(card_type) 
                  for card_type in self.TOTAL_CARDS)

class GameState:
    """游戏状态类"""
    def __init__(self):
        self.cards = []  # 当前场上的所有卡牌
        self.slot_cards = []  # 底部槽中的卡牌
        self.power = 0  # 闪电能量值
        self.shuffle_count = 0  # 洗牌道具数量
        self.remove_count = 0  # 移出道具数量
        self.undo_count = 0  # 撤回道具数量
        self.card_stats = CardStats()  # 卡牌统计
        
    def add_card(self, card: Card):
        """添加卡牌到游戏状态"""
        self.cards.append(card)
        status = 'visible' if card.is_visible else 'hidden'
        self.card_stats.update_card_count(card.type, status)
        
    def remove_card(self, card: Card):
        """从游戏状态中移除卡牌"""
        if card in self.cards:
            self.cards.remove(card)
            status = 'visible' if card.is_visible else 'hidden'
            self.card_stats.update_card_count(card.type, status, -1)
            self.card_stats.update_card_count(card.type, 'eliminated')
            
    def add_to_slot(self, card: Card):
        """添加卡牌到底部槽"""
        if len(self.slot_cards) < 7:  # 假设槽最多容纳7张卡牌
            self.slot_cards.append(card)
            self.card_stats.update_card_count(card.type, 'slot')
            return True
        return False
        
    def get_available_cards(self) -> List[Card]:
        """获取当前可点击的卡牌"""
        return [card for card in self.cards if card.is_visible]
    
    def get_matching_cards(self, card_type: str) -> List[Card]:
        """获取指定类型的可配对卡牌"""
        return [card for card in self.get_available_cards() 
                if card.type == card_type]
    
    def get_best_matches(self) -> List[Tuple[str, float]]:
        """获取最佳配对选择及其概率"""
        matches = []
        for card_type in self.card_stats.TOTAL_CARDS:
            prob = self.card_stats.get_matching_probability(card_type)
            if prob > 0:
                matches.append((card_type, prob))
        # 按概率降序排序
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def can_use_power(self) -> bool:
        """检查是否可以使用能量"""
        return self.power > 0
    
    def use_power(self):
        """使用一点能量"""
        if self.can_use_power():
            self.power -= 1
            return True
        return False
    
    def to_feature_vector(self) -> np.ndarray:
        """将游戏状态转换为特征向量，用于机器学习"""
        features = []
        
        # 添加卡牌状态
        card_grid = np.zeros((10, 10))  # 假设游戏区域为10x10网格
        for card in self.cards:
            x, y = card.position
            card_grid[x, y] = 1 if card.is_visible else 0.5
            
        # 添加槽位状态
        slot_state = np.zeros(7)
        for i, card in enumerate(self.slot_cards):
            slot_state[i] = 1
            
        # 添加道具和能量状态
        game_state = [
            self.power / 10.0,  # 归一化能量值
            self.shuffle_count / 5.0,  # 归一化道具数量
            self.remove_count / 5.0,
            self.undo_count / 5.0
        ]
        
        # 添加卡牌统计信息
        for card_type in self.card_stats.TOTAL_CARDS:
            features.extend([
                self.card_stats.remaining_cards[card_type] / self.card_stats.TOTAL_CARDS[card_type],
                self.card_stats.visible_cards[card_type] / self.card_stats.TOTAL_CARDS[card_type],
                self.card_stats.get_matching_probability(card_type)
            ])
        
        # 合并所有特征
        features = np.concatenate([
            card_grid.flatten(),
            slot_state,
            np.array(game_state),
            np.array(features)
        ])
        
        return features 