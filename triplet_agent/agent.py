# -*- encoding: utf-8 -*-
import time
import logging
from typing import Dict, Any, Optional, List, Set

from .configs import DEFAULT_CFG, FULL_TABLE_KEYWORDS
from .planner import FeatureDetector
from .router import AgentRouter
from .memory import DialogState
from .tools.filter_tool import TripletFilter, match_question_entities, merge_unique, expand_neighbors
from .tools.render_tool import ContextRenderer
from .qwen_client import QwenHttpClient

_GUIDANCE_MAP = {
    "group": "先理解行层级与归属关系，再分步作答。",
    "comparison": "明确比较对象与指标列，列出差异点。",
    "statistics": "区分明细行与汇总行，列出算式确保准确。",
    "correlation": "观察两列数值在不同行中的变化趋势是否一致。",
    "causal": "通过指标变动解释原因。",
    "trend": "按时间顺序对比数据，并说明趋势。",
    "anomaly": "指出偏离正常范围的离群数值。",
}

class TripletAgent:
    """SOTA V10+ 特征触发制 Agent：终极对齐实验逻辑版。"""

    def __init__(self, client, qwen_url: Optional[str] = None, max_retries: int = 5):
        self.client = client
        self.max_retries = max_retries
        self.detector = FeatureDetector()
        self.router = AgentRouter()
        self.filter_tool = TripletFilter(client)
        self.renderer = ContextRenderer()
        self.qwen_client = QwenHttpClient(base_url=qwen_url) if qwen_url else None

    def process_triplet_query(
        self,
        triplet_data: Dict[str, Any],
        question_list: List[str],
        instruction_template: str,
        system_message: str = "You are a helpful assistant.",
        pre_context: str = "",
        post_context: str = "",
        route_mode: str = "auto",
        learnable_router: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        route_mode:
          - "auto"     : 当前 SOTA V5 规则路由（默认，保持原行为）
          - "full"     : 强制 path C 全量推理
          - "partial"  : 强制 path B + 保留覆盖率回退
          - "filter"   : 强制 path B + 关闭覆盖率回退（最强筛选）
          - "learnable": 调用 learnable_router.predict_route(...) 决定 full/partial/filter
        """
        prediction_list = []
        per_turn_route_mode: List[str] = []
        per_turn_input_tokens: List[int] = []
        group = triplet_data.get("group") or []
        real_items, real_features = self.renderer._extract_group_keys(group)
        table_cells = len(real_items) * len(real_features)

        history, dialog_state = [], DialogState()
        is_multi_turn = len(question_list) > 1

        for question in question_list:
            is_first_turn = len(history) == 0
            features = self.detector.detect(question, qwen_client=self.qwen_client, is_first_turn=is_first_turn)

            # 直答捷径优先级最高，所有 route_mode 下都保留
            if self.router.is_direct_answer_question(question):
                direct_ans = f"行数：{len(real_items)}，列数：{len(real_features)}"
                prediction_list.append(direct_ans)
                per_turn_route_mode.append("direct")
                per_turn_input_tokens.append(0)
                history.append((question, direct_ans))
                dialog_state.update(question, real_items, real_features, match_fn=match_question_entities)
                continue

            # 决定本轮 route_mode（learnable 在此处解析为 full/partial/filter）
            turn_mode = route_mode
            if turn_mode == "learnable":
                if learnable_router is None:
                    raise ValueError("route_mode='learnable' requires a learnable_router instance")
                meta = {
                    "rows": len(real_items),
                    "cols": len(real_features),
                    "cells": table_cells,
                    "lang": triplet_data.get("lang", "zh"),
                }
                turn_mode = learnable_router.predict_route(question, meta)
                if turn_mode not in {"full", "partial", "filter"}:
                    raise ValueError(f"learnable_router returned invalid route_mode: {turn_mode!r}")
            per_turn_route_mode.append(turn_mode)

            full_context_est = self.renderer.triplets_to_markdown(triplet_data, pre_context, post_context)
            force_filter = (len(full_context_est) // 1.2) > 2000
            force_full_first_turn = is_first_turn and is_multi_turn

            decision = "full"
            sel_items, sel_features = [], []

            # 路由决策：强制覆盖 or auto 规则
            forced = AgentRouter.force_filter_decision(turn_mode)
            if forced is None:
                try_filter = self.router.should_try_filter(
                    question, features, table_cells, real_items, real_features
                )
            else:
                try_filter = forced

            if (not force_full_first_turn and (force_filter or try_filter)):
                state_summary = dialog_state.build_summary()
                decision, sel_items, sel_features, _ = self.filter_tool.llm_filter_with_state(
                    question, real_items, real_features, state_summary
                )

                if decision == "filter":
                    lexical_items = match_question_entities(question, real_items, limit=8)
                    sel_items = merge_unique(sel_items, lexical_items)

                    is_follow_up = not is_first_turn
                    if "referential" in features or is_follow_up or not sel_items:
                        sel_items = merge_unique(sel_items, dialog_state.get_items())
                        sel_features = merge_unique(sel_features, dialog_state.get_features())

                    sel_items = expand_neighbors(sel_items, real_items, window=1)

                    # 覆盖率熔断：auto/partial 保留；filter 强制关掉
                    keep_fallback = AgentRouter.should_keep_coverage_fallback(turn_mode)
                    if keep_fallback and not force_filter and len(real_items) > 0 and \
                       len(sel_items) / len(real_items) >= DEFAULT_CFG.coverage_fallback:
                        decision = "full"
                        sel_items, sel_features = [], []

                if force_filter and decision != "filter":
                    decision = "filter"
                    sel_items = match_question_entities(question, real_items, limit=12)
                    sel_features = match_question_entities(question, real_features, limit=12)
                    if not sel_items: sel_items = dialog_state.get_items() or real_items[:8]
                    if not sel_features: sel_features = dialog_state.get_features() or real_features[:10]

            render_task = "group" if "group" in features else ""
            context = self.renderer.triplets_to_markdown(
                triplet_data, pre_context, post_context,
                selected_items=sel_items if decision == "filter" else None,
                selected_features=sel_features if decision == "filter" else None,
                sub_task_name=render_task
            )

            messages = [{"role": "system", "content": system_message}]
            for hq, ha in history:
                messages.append({"role": "user", "content": hq})
                messages.append({"role": "assistant", "content": ha})

            if is_first_turn:
                user_input = instruction_template.replace("{context}", context).replace("{question}", question)
                user_input += self._build_sota_guidance(features)
            elif decision == "filter":
                user_input = f"补充参考数据：\n{context}\n\n问题：{question}"
            else:
                user_input = instruction_template.replace("{context}", context).replace("{question}", question)

            messages.append({"role": "user", "content": user_input})
            model_response, usage = self._generate_with_retry_and_usage(messages)
            prediction_list.append(model_response)
            per_turn_input_tokens.append(int(usage.get("input_tokens", 0)) if usage else 0)
            history.append((question, model_response))
            dialog_state.update(question, real_items, real_features, sel_items, sel_features, match_fn=match_question_entities)

        return {
            "prediction_list": prediction_list,
            "route_mode_per_turn": per_turn_route_mode,
            "input_tokens_per_turn": per_turn_input_tokens,
        }

    def _build_sota_guidance(self, features: Set[str]) -> str:
        hints = [text for feat, text in _GUIDANCE_MAP.items() if feat in features]
        return "\n\n补充要求：\n" + "\n".join(f"{idx+1}. {h}" for idx, h in enumerate(hints)) if hints else ""

    def _generate_with_retry(self, messages: List[Dict]) -> str:
        # 旧调用方兼容：丢掉 usage
        response, _ = self._generate_with_retry_and_usage(messages)
        return response

    def _generate_with_retry_and_usage(self, messages: List[Dict]):
        for _ in range(self.max_retries):
            response, usage = self.client.generate_response(messages)
            if response: return response, (usage or {})
            time.sleep(5)
        return "Error: Failed to generate response.", {}
