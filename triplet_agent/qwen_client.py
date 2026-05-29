# -*- encoding: utf-8 -*-
import requests
import json
import logging
from typing import List, Set, Dict, Any

class QwenHttpClient:
    """
    Qwen 专用轻量化客户端，用于特征探测 (8001-8003 端口负载均衡或单点)。
    """
    def __init__(self, base_url: str = "http://localhost:8001/v1"):
        self.base_url = base_url

    def detect_features(self, question: str) -> Set[str]:
        """调用 Qwen 进行语义特征识别。"""
        prompt = (
            "分析以下问题，判断其属于哪些特征。可多选：\n"
            "- group: 分组、分类、按类别汇总\n"
            "- statistics: 统计、求和、平均、计数、占比\n"
            "- comparison: 对比、比较、排名、排序、最大最小、增长下降\n"
            "- causal: 因果分析、解释原因\n"
            "- trend: 趋势分析、变化规律\n"
            "- correlation: 相关性判断\n"
            "- anomaly: 异常检测\n\n"
            f"问题：{question}\n\n"
            '返回 JSON 数组，如 ["statistics", "comparison"]，不要输出其它内容。'
        )
        
        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=10)
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            
            # 解析 JSON 数组
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                features = json.loads(content[start:end+1])
                return set(features)
        except Exception as e:
            logging.error(f"Qwen detect_features failed: {e}")
            
        return set()
