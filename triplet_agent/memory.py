# -*- encoding: utf-8 -*-
from typing import List, Set, Optional, Callable

class DialogState:
    """
    Agent 记忆槽位：跨轮追踪关键实体。
    用于在多轮对话中，将第一轮识别到的 items/features 传递给后续轮。
    """
    def __init__(self):
        self.last_items: List[str] = []
        self.last_features: List[str] = []
        self.turn_count = 0

    def update(
        self,
        question: str,
        all_items: List[str],
        all_features: List[str],
        current_items: Optional[List[str]] = None,
        current_features: Optional[List[str]] = None,
        match_fn: Optional[Callable] = None
    ):
        self.turn_count += 1
        
        # 1. 如果显式传入了当前选中的，直接更新
        if current_items is not None:
            self.last_items = list(dict.fromkeys(self.last_items + current_items))
        if current_features is not None:
            self.last_features = list(dict.fromkeys(self.last_features + current_features))
            
        # 2. 如果提供了匹配函数，自动从问题中提取补充
        if match_fn:
            mi = match_fn(question, all_items, limit=8)
            if mi:
                self.last_items = list(dict.fromkeys(self.last_items + mi))
            mf = match_fn(question, all_features, limit=8)
            if mf:
                self.last_features = list(dict.fromkeys(self.last_features + mf))

    def get_items(self) -> List[str]:
        return self.last_items

    def get_features(self) -> List[str]:
        return self.last_features

    def build_summary(self) -> str:
        if not self.last_items and not self.last_features:
            return ""
        return f"前文提及实体: {', '.join(self.last_items[:5])}; 提及特征: {', '.join(self.last_features[:5])}"
