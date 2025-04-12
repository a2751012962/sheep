<<<<<<< HEAD
from typing import List, Tuple, Optional
from image_recognition.game_state import GameState, Card

class Strategy:
    """策略生成器类"""
    def __init__(self, allow_items: bool = True):
        self.last_action = None
        self.action_history = []
        self.allow_items = allow_items  # 是否允许使用道具
        
    def set_item_usage(self, allow: bool):
        """设置是否允许使用道具"""
        self.allow_items = allow
        
    def generate_action(self, game_state: GameState) -> Tuple[str, Optional[Card]]:
        """生成下一步动作"""
        # 检查是否有可以直接配对的卡牌
        direct_matches = self._find_direct_matches(game_state)
        if direct_matches:
            return 'match', direct_matches[0]
            
        # 检查是否需要使用道具（仅当允许使用道具时）
        if self.allow_items and self._should_use_item(game_state):
            item_action = self._choose_best_item(game_state)
            if item_action:
                return item_action, None
                
        # 选择最有潜力的卡牌
        best_card = self._choose_best_potential_card(game_state)
        if best_card:
            return 'select', best_card
            
        # 如果实在没有选择且允许使用道具，尝试使用道具
        if self.allow_items and game_state.slot_cards:
            for item in ['shuffle', 'remove', 'undo']:
                if getattr(game_state, f"{item}_count", 0) > 0:
                    return item, None
            
        return 'wait', None
        
    def _find_direct_matches(self, game_state: GameState) -> List[Card]:
        """寻找可以直接配对的卡牌"""
        available_cards = game_state.get_available_cards()
        card_counts = {}
        
        # 统计每种类型的可用卡牌数量
        for card in available_cards:
            card_counts[card.type] = card_counts.get(card.type, []) + [card]
            
        # 返回数量大于等于3的卡牌
        for cards in card_counts.values():
            if len(cards) >= 3:
                return cards[:3]
        return []
        
    def _should_use_item(self, game_state: GameState) -> bool:
        """判断是否应该使用道具"""
        if not self.allow_items:  # 如果不允许使用道具，直接返回False
            return False
            
        # 如果槽快满了，考虑使用道具
        if len(game_state.slot_cards) >= 6:
            return True
            
        # 检查槽中的卡牌是否有可能完成组合
        slot_types = {}
        for card in game_state.slot_cards:
            slot_types[card.type] = slot_types.get(card.type, 0) + 1
            
        # 如果有超过2张相同类型的卡牌，但无法凑成3张，考虑使用道具
        for card_type, count in slot_types.items():
            if count == 2:
                remaining = game_state.card_stats.remaining_cards[card_type]
                if remaining == 0:
                    return True
                    
        # 如果没有可配对的牌，且匹配概率都很低，考虑使用道具
        best_matches = game_state.get_best_matches()
        if not best_matches or best_matches[0][1] < 0.1:
            # 检查是否还有足够的组数
            total_groups = game_state.card_stats.get_total_groups()
            if total_groups < len(game_state.slot_cards) / 3:
                return True
            
        return False
        
    def _choose_best_item(self, game_state: GameState) -> Optional[str]:
        """选择最适合使用的道具"""
        if not self.allow_items:  # 如果不允许使用道具，直接返回None
            return None
            
        if game_state.shuffle_count > 0:
            # 如果槽中有多张相同类型的牌，优先使用洗牌
            slot_types = {}
            for card in game_state.slot_cards:
                slot_types[card.type] = slot_types.get(card.type, 0) + 1
                
            # 如果有2张相同类型的卡牌，但剩余卡牌不足以凑成一组，使用洗牌
            for card_type, count in slot_types.items():
                if count == 2 and game_state.card_stats.remaining_cards[card_type] == 0:
                    return 'shuffle'
                    
        if game_state.remove_count > 0:
            # 如果槽中有单张无法配对的牌，使用移出
            for card in game_state.slot_cards:
                remaining = game_state.card_stats.remaining_cards[card.type]
                remaining_groups = game_state.card_stats.get_remaining_groups(card.type)
                if remaining_groups == 0:
                    return 'remove'
                    
        if game_state.undo_count > 0:
            # 如果最后一步操作导致槽接近满了，且无法形成组合，考虑撤回
            if len(game_state.slot_cards) >= 6 and self.last_action:
                slot_types = {}
                for card in game_state.slot_cards:
                    slot_types[card.type] = slot_types.get(card.type, 0) + 1
                    
                # 检查是否有可能的组合
                has_potential_group = False
                for count in slot_types.values():
                    if count >= 2:
                        has_potential_group = True
                        break
                        
                if not has_potential_group:
                    return 'undo'
                    
        return None
        
    def _choose_best_potential_card(self, game_state: GameState) -> Optional[Card]:
        """选择最有潜力的卡牌"""
        available_cards = game_state.get_available_cards()
        if not available_cards:
            return None
            
        # 获取每种卡牌的评分
        card_scores = {}
        for card in available_cards:
            # 基础匹配概率
            prob = game_state.card_stats.get_matching_probability(card.type)
            
            # 剩余组数因子（优先选择剩余组数较多的卡牌）
            remaining_groups = game_state.card_stats.get_remaining_groups(card.type)
            groups_factor = remaining_groups / 5.0  # 归一化，假设最多5组
            
            # 槽中已有卡牌加成
            slot_bonus = sum(1 for c in game_state.slot_cards if c.type == card.type)
            
            # 计算综合得分
            score = (
                prob * 0.4 +              # 匹配概率权重
                groups_factor * 0.4 +     # 剩余组数权重
                slot_bonus * 0.2          # 槽中已有卡牌权重
            )
            
            card_scores[card] = score
            
        # 返回得分最高的卡牌
        return max(card_scores.items(), key=lambda x: x[1])[0]
        
    def update_history(self, action: str, card: Optional[Card], success: bool):
        """更新操作历史"""
        self.last_action = (action, card, success)
        self.action_history.append(self.last_action)
        
        # 只保留最近的100步操作历史
        if len(self.action_history) > 100:
=======
from typing import List, Tuple, Optional
from image_recognition.game_state import GameState, Card

class Strategy:
    """策略生成器类"""
    def __init__(self, allow_items: bool = True):
        self.last_action = None
        self.action_history = []
        self.allow_items = allow_items  # 是否允许使用道具
        
    def set_item_usage(self, allow: bool):
        """设置是否允许使用道具"""
        self.allow_items = allow
        
    def generate_action(self, game_state: GameState) -> Tuple[str, Optional[Card]]:
        """生成下一步动作"""
        # 检查是否有可以直接配对的卡牌
        direct_matches = self._find_direct_matches(game_state)
        if direct_matches:
            return 'match', direct_matches[0]
            
        # 检查是否需要使用道具（仅当允许使用道具时）
        if self.allow_items and self._should_use_item(game_state):
            item_action = self._choose_best_item(game_state)
            if item_action:
                return item_action, None
                
        # 选择最有潜力的卡牌
        best_card = self._choose_best_potential_card(game_state)
        if best_card:
            return 'select', best_card
            
        # 如果实在没有选择且允许使用道具，尝试使用道具
        if self.allow_items and game_state.slot_cards:
            for item in ['shuffle', 'remove', 'undo']:
                if getattr(game_state, f"{item}_count", 0) > 0:
                    return item, None
            
        return 'wait', None
        
    def _find_direct_matches(self, game_state: GameState) -> List[Card]:
        """寻找可以直接配对的卡牌"""
        available_cards = game_state.get_available_cards()
        card_counts = {}
        
        # 统计每种类型的可用卡牌数量
        for card in available_cards:
            card_counts[card.type] = card_counts.get(card.type, []) + [card]
            
        # 返回数量大于等于3的卡牌
        for cards in card_counts.values():
            if len(cards) >= 3:
                return cards[:3]
        return []
        
    def _should_use_item(self, game_state: GameState) -> bool:
        """判断是否应该使用道具"""
        if not self.allow_items:  # 如果不允许使用道具，直接返回False
            return False
            
        # 如果槽快满了，考虑使用道具
        if len(game_state.slot_cards) >= 6:
            return True
            
        # 检查槽中的卡牌是否有可能完成组合
        slot_types = {}
        for card in game_state.slot_cards:
            slot_types[card.type] = slot_types.get(card.type, 0) + 1
            
        # 如果有超过2张相同类型的卡牌，但无法凑成3张，考虑使用道具
        for card_type, count in slot_types.items():
            if count == 2:
                remaining = game_state.card_stats.remaining_cards[card_type]
                if remaining == 0:
                    return True
                    
        # 如果没有可配对的牌，且匹配概率都很低，考虑使用道具
        best_matches = game_state.get_best_matches()
        if not best_matches or best_matches[0][1] < 0.1:
            # 检查是否还有足够的组数
            total_groups = game_state.card_stats.get_total_groups()
            if total_groups < len(game_state.slot_cards) / 3:
                return True
            
        return False
        
    def _choose_best_item(self, game_state: GameState) -> Optional[str]:
        """选择最适合使用的道具"""
        if not self.allow_items:  # 如果不允许使用道具，直接返回None
            return None
            
        if game_state.shuffle_count > 0:
            # 如果槽中有多张相同类型的牌，优先使用洗牌
            slot_types = {}
            for card in game_state.slot_cards:
                slot_types[card.type] = slot_types.get(card.type, 0) + 1
                
            # 如果有2张相同类型的卡牌，但剩余卡牌不足以凑成一组，使用洗牌
            for card_type, count in slot_types.items():
                if count == 2 and game_state.card_stats.remaining_cards[card_type] == 0:
                    return 'shuffle'
                    
        if game_state.remove_count > 0:
            # 如果槽中有单张无法配对的牌，使用移出
            for card in game_state.slot_cards:
                remaining = game_state.card_stats.remaining_cards[card.type]
                remaining_groups = game_state.card_stats.get_remaining_groups(card.type)
                if remaining_groups == 0:
                    return 'remove'
                    
        if game_state.undo_count > 0:
            # 如果最后一步操作导致槽接近满了，且无法形成组合，考虑撤回
            if len(game_state.slot_cards) >= 6 and self.last_action:
                slot_types = {}
                for card in game_state.slot_cards:
                    slot_types[card.type] = slot_types.get(card.type, 0) + 1
                    
                # 检查是否有可能的组合
                has_potential_group = False
                for count in slot_types.values():
                    if count >= 2:
                        has_potential_group = True
                        break
                        
                if not has_potential_group:
                    return 'undo'
                    
        return None
        
    def _choose_best_potential_card(self, game_state: GameState) -> Optional[Card]:
        """选择最有潜力的卡牌"""
        available_cards = game_state.get_available_cards()
        if not available_cards:
            return None
            
        # 获取每种卡牌的评分
        card_scores = {}
        for card in available_cards:
            # 基础匹配概率
            prob = game_state.card_stats.get_matching_probability(card.type)
            
            # 剩余组数因子（优先选择剩余组数较多的卡牌）
            remaining_groups = game_state.card_stats.get_remaining_groups(card.type)
            groups_factor = remaining_groups / 5.0  # 归一化，假设最多5组
            
            # 槽中已有卡牌加成
            slot_bonus = sum(1 for c in game_state.slot_cards if c.type == card.type)
            
            # 计算综合得分
            score = (
                prob * 0.4 +              # 匹配概率权重
                groups_factor * 0.4 +     # 剩余组数权重
                slot_bonus * 0.2          # 槽中已有卡牌权重
            )
            
            card_scores[card] = score
            
        # 返回得分最高的卡牌
        return max(card_scores.items(), key=lambda x: x[1])[0]
        
    def update_history(self, action: str, card: Optional[Card], success: bool):
        """更新操作历史"""
        self.last_action = (action, card, success)
        self.action_history.append(self.last_action)
        
        # 只保留最近的100步操作历史
        if len(self.action_history) > 100:
>>>>>>> d1b411e347c1dbf5d2d30dbf0828bd283efb0dec
            self.action_history.pop(0) 