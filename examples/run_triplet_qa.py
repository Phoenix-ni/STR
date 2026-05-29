# -*- encoding: utf-8 -*-
import sys
import os
from typing import List, Dict, Any, Tuple

# 将父目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triplet_agent.agent import TripletAgent

class MockLLMClient:
    """示例大模型客户端包装"""
    def generate_response(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
        # 这里集成你的实际 LLM 调用，如 OpenAI, DashScope 等
        print(f"\n[LLM Request] {messages[-1]['content'][:100]}...")
        return "这是模拟的回答内容。", {"input_tokens": 100, "output_tokens": 50}

def main():
    # 1. 模拟数据
    triplet_data = {
        "shape": "3x2",
        "group": [
            {"item": "公司A", "feature": "收入", "value": "100"},
            {"item": "公司A", "feature": "利润", "value": "20"},
            {"item": "公司B", "feature": "收入", "value": "200"},
            {"item": "公司B", "feature": "利润", "value": "50"},
            {"item": "公司C", "feature": "收入", "value": "150"},
            {"item": "公司C", "feature": "利润", "value": "30"},
        ]
    }
    
    questions = ["哪家公司的收入最高？", "那它的利润呢？"]
    instruction = "请根据以下表格内容回答问题。\n\n表格内容：\n{context}\n\n问题：{question}"
    
    # 2. 初始化 Agent
    client = MockLLMClient()
    agent = TripletAgent(client)
    
    # 3. 执行查询
    print("开始处理问题列表...")
    result = agent.process_triplet_query(
        triplet_data=triplet_data,
        question_list=questions,
        instruction_template=instruction
    )
    
    # 4. 输出结果
    for q, a in zip(questions, result["prediction_list"]):
        print(f"\n问：{q}")
        print(f"答：{a}")
    
    print(f"\n路由: {result['route_mode_per_turn']}")
    print(f"输入 tokens: {result['input_tokens_per_turn']}")

if __name__ == "__main__":
    main()
