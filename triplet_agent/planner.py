# -*- encoding: utf-8 -*-
"""
Feature Detector — 专精分流架构。
核心逻辑：
1. 优先正则预判：点查询直接短路（不调 Qwen）。
2. 高价值触发：仅在检测到 分组、统计、因果、拒答 信号时，才调用 Qwen 进行语义确认。
3. 极致降费：让 Qwen 只出现在它能产生高价值增量的子任务上。
"""
import re
import logging
from typing import Set, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen_client import QwenHttpClient

# ── 特征 → 正则规则 ──────────────────────────────────────────────────

_FEATURE_RULES = {
    # 1. 点查询白名单（易识别，正则极其稳健）
    "simple": [
        (re.compile(r"(?:是多少|是啥|是哪个|是哪家|是哪一个|是哪项|归属于哪个|属于哪|哪些|哪几)"), None),
        (re.compile(r"(?:为何值|具体是|在(?:哪里|哪儿|什么位置|哪个位置))"), None),
        (re.compile(r"(?:是什么|请列出|告诉我|具体).{0,5}(?:名称|项目|日期|金额|数值|内容|公司|国家|组织)"), None),
        (re.compile(r"(?:谁是|谁的|哪个是|哪个项目|哪个时间|哪些项目|哪些公司|哪些模型)"), None),
        (re.compile(r"第[一二三四五六七八九十0-9]+行"), None),
    ],

    # 2. 高价值触发信号（需要 Qwen 介入以提升准确率的 4 类任务）
    "high_value_trigger": [
        # 分组查询
        (re.compile(r"(?:划分|分组|分类|归类|按.*分为|类别)"), None),
        # 统计分析
        (re.compile(r"(?:统计|总计|总共|一共|共有|合计|总和|求和|平均值?|均值)"), None),
        # 因果分析
        (re.compile(r"(?:原因|为什么|为何|理由|因为)"), None),
        # 拒答/存在性（语义性强，正则难识别，建议调 Qwen）
        (re.compile(r"(?:是否|有没有|有没有提到|是否存在|记录)"), None),
    ],

    # 3. 其他需要全表视野但正则可识别的辅助特征
    "needs_full": [
        (re.compile(r"(?:所有|每一|全部|全表)"), None),
        (re.compile(r"(?:排名|排序|最高|最低|最大|最小|对比|差异|最好|最差|较好|较差)"), None),
    ],

    # 元数据直答
    "direct_answer": [
        (re.compile(r"(?:多少|几)(?:行|列)"), re.compile(r"(?:多少个|多少家|多少种|多少次|多少项|多少只|多少位)")),
        (re.compile(r"(?:行数|列数|行列|表格大小|表格的大小)"), None),
        # "最大行/最大列" —— "该表格的最大行和最大列是多少？"
        (re.compile(r"最大行|最大列"), None),
        # "行和列/行与列 + 数量/是多少/多少/大小"
        # —— "所示表格的行和列的数量是多少？"、"表格行列的最大大小是多少？"
        (re.compile(r"行[和与].{0,2}列.{0,8}(?:数量|是多少|多少|数|大小|大小是)"), None),
    ],
    
    # 合并单元格
    "merge_cell": [
        (re.compile(r"合并(?:的|了)?(?:单元格|格子|cells?)"), None),
    ],

    # 代指判断（严格模式：仅保留强信号，规避“它”、“该”等高频词误解）
    # 扩充明确指向前文的短语：如上/上面/刚才 等在首轮也不会误伤
    "referential": [
        (re.compile(r"(?:上述|前者|后者|前面提到|它的|它们|如上|上面|刚才|刚刚提到|前述)"), None),
    ],
}


class FeatureDetector:
    """
    分级分流提取器。
    逻辑：Simple + 不是 High_Value => 直接短路。
    否则：调用 Qwen 辅助判断语义增量。
    """

    def detect(self, question: str, qwen_client: Optional['QwenHttpClient'] = None,
               is_first_turn: bool = True) -> Set[str]:
        q = question.strip()
        features: Set[str] = set()

        # 1. 快速正则一次扫描
        matched_groups = {name: False for name in _FEATURE_RULES.keys()}
        for name, patterns in _FEATURE_RULES.items():
            for match_re, exclude_re in patterns:
                if match_re.search(q):
                    if exclude_re is not None and exclude_re.search(q):
                        continue
                    features.add(name)
                    matched_groups[name] = True
                    break

        # 2. 正则强约束：referential 命中 → 自动提升 needs_full
        # Qwen 后续只能 union features（无法降级），从而保住全表保护不被 Qwen 冲掉
        if matched_groups["referential"]:
            features.add("needs_full")

        # 3. 分流决策
        # 短路判定：如果是点查询，且没有任何高价值信号，也没有复杂的全局关键词 -> SKIP QWEN
        is_simple_lookup = matched_groups["simple"] or matched_groups["direct_answer"]
        has_complex_signal = matched_groups["high_value_trigger"] or matched_groups["needs_full"] or matched_groups["merge_cell"]

        # 核心逻辑：只有在【不是简单查询】或者【命中了高价值触发词】时，才需要 Qwen 介入
        should_call_qwen = has_complex_signal or (not is_simple_lookup)

        # 4. Qwen 协助分流（专精于提升 分组、统计、因果、拒答 的准确率）
        # 注意：Qwen 结果只做 union，无法移除正则已标记的 needs_full/referential
        # 保留多轮 follow-up 的 Qwen 调用：对比分析等依赖 Qwen 注入 comparison 等特征的指导语
        if qwen_client and should_call_qwen:
            try:
                # 只在这些高增量场景下调用
                qwen_features = qwen_client.detect_features(question)
                features.update(qwen_features)
            except Exception as e:
                logging.error(f"Qwen select-routing failed: {e}")
        elif is_simple_lookup:
            logging.info(f"Routing: [Regex Short-circuit] for query: {question[:30]}...")

        return features
