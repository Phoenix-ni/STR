# -*- encoding: utf-8 -*-
from typing import List, Set, Optional
from .configs import (
    DEFAULT_CFG, DIRECT_ANSWER_PATTERNS, DIRECT_ANSWER_NOISY_PATTERNS,
    FULL_TABLE_KEYWORDS
)
from .tools.filter_tool import match_question_entities

# Learnable routing 三档强度 + auto = 用规则路由（当前 SOTA V5 行为）。
# learnable = 由外部传入的 learnable_router.predict_route 决定，但最终落到
# 这四个标签之一交给 agent.py 执行。
ROUTE_MODES = ("auto", "full", "partial", "filter", "learnable")


class AgentRouter:
    """SOTA 路由逻辑：严格对齐实验中的 V5 分流策略。"""

    def is_direct_answer_question(self, question: str) -> bool:
        q = question.strip()
        if not any(p in q for p in DIRECT_ANSWER_PATTERNS):
            return False
        return not any(p in q for p in DIRECT_ANSWER_NOISY_PATTERNS)

    def should_try_filter(
        self,
        question: str,
        features: Set[str],
        table_cells: int,
        items: List[str],
        features_list: List[str]
    ) -> bool:
        """严格复刻实验中的 _should_try_filter 函数。"""
        # 1. 小表不滤 (阈值 60)
        if table_cells <= 60:
            return False
        # 2. 强信号特征拦截
        if "needs_full" in features:
            return False
        # 3. 全表关键词拦截
        if any(kw in question for kw in FULL_TABLE_KEYWORDS):
            return False
        # 4. 指代/省略问题的特殊处理
        if "referential" in features:
            if match_question_entities(question, items, limit=1) or \
               match_question_entities(question, features_list, limit=1):
                return True
            return False
        # 5. 必须在问题中找到具体实体引用才允许过滤
        if match_question_entities(question, items, limit=1):
            return True
        if match_question_entities(question, features_list, limit=1):
            return True
        return False

    @staticmethod
    def force_filter_decision(route_mode: str) -> Optional[bool]:
        """
        Learnable routing 接口：把强制路由模式翻译为 should_try_filter 的返回。

        返回 None  → 让调用方继续走 auto 规则
        返回 True  → 强制进 filter 分支（path B）
        返回 False → 强制走 path C 全量推理
        """
        if route_mode not in {"auto", "full", "partial", "filter"}:
            raise ValueError(f"route_mode must be one of auto/full/partial/filter, got {route_mode!r}")
        if route_mode == "auto":
            return None
        if route_mode == "full":
            return False
        # partial / filter 都走 filter 分支，差异由 should_keep_coverage_fallback 决定
        return True

    @staticmethod
    def should_keep_coverage_fallback(route_mode: str) -> bool:
        """
        partial / auto = 保留覆盖率熔断（如果选中过多 item 就回退 full）
        filter         = 强制不回退，强筛选到底
        """
        return route_mode in ("auto", "partial")
