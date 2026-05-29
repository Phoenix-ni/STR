# -*- encoding: utf-8 -*-
from dataclasses import dataclass

# ── 核心常量 ──────────────────────────────────────────────────────────

TRIPLET_FORMAT_EXPLANATION = (
    "以下是表格的三元组(Triplet)表示：\n"
    "- item（条目）：行标签\n"
    "- feature（特征）：列标签\n"
    "- value（值）：交叉点内容\n"
    "下方表格第一列为 item，表头行为 feature。"
)

DIRECT_ANSWER_PATTERNS = ["多少行", "多少列", "表格大小", "行数", "列数", "行列数"]
DIRECT_ANSWER_NOISY_PATTERNS = ["多少个", "多少家", "多少种", "多少次", "多少项"]

# SOTA V5 路由核心关键词
FULL_TABLE_KEYWORDS = [
    "列出所有", "所有", "哪些", "哪几", "分别", "统计", "总计", "总和",
    "求和", "平均", "均值", "排名", "排序", "分组", "趋势", "变化",
    "比较", "对比", "占比", "比例", "最大", "最小", "最高", "最低",
]

REFERENTIAL_PATTERNS_STRICT = ["它的", "它们", "上述", "前者", "后者", "前面提到", "如上", "上面"]

@dataclass(frozen=True)
class OracleConfig:
    """SOTA V5 策略参数"""
    tiny: int = 60              # 小于 60 格强制全量
    coverage_fallback: float = 0.75  # 过滤覆盖率 > 75% 回退全量
    dual_coverage_fallback: bool = True
    
# 默认采用 SOTA 配置
DEFAULT_CFG = OracleConfig()
