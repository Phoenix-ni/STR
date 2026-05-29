"""
TEDS (Tree-Edit-Distance-based Similarity) 评估指标

对应论文 Section 4.1：
    - TEDS: 完整指标（含单元格内容）
    - TEDS-Struc: 仅结构（不含内容）
    - Accuracy: 完全正确预测的表格比例

TEDS 最初由 EDD [57] 提出：
    Zhong et al. "Image-based Table Recognition: Data, Model, and Evaluation" (ECCV 2020)

公式：TEDS(T_a, T_b) = 1 - EditDist(T_a, T_b) / max(|T_a|, |T_b|)
"""

import re
from typing import List, Dict, Tuple, Optional
from html.parser import HTMLParser
from apted import APTED, Config
from apted.helpers import Tree


class TableNode:
    """
    HTML 表格解析树节点

    用于构建树形结构，供 APTED 算法计算编辑距离
    """

    def __init__(self, tag: str, attrs: dict = None):
        self.tag = tag
        self.attrs = attrs or {}
        self.children = []
        self.text = ''

    def bracket(self) -> str:
        """转换为 APTED bracket 格式"""
        # 节点标签（用于树编辑距离比较）
        node_label = self._get_label()
        children_str = ''.join(c.bracket() for c in self.children)
        return f'{{{node_label}{children_str}}}'

    def _get_label(self) -> str:
        """获取节点标签（用于编辑距离比较）"""
        if self.tag in ('td', 'th'):
            # 包含 rowspan/colspan 属性
            rs = self.attrs.get('rowspan', '1')
            cs = self.attrs.get('colspan', '1')
            return f'{self.tag} rowspan={rs} colspan={cs}'
        return self.tag


class HTMLTableParser(HTMLParser):
    """HTML 表格解析器，将 HTML 转换为树结构"""

    def __init__(self):
        super().__init__()
        self.root = TableNode('root')
        self.stack = [self.root]
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag not in ('table', 'tr', 'td', 'th', 'thead', 'tbody', 'tfoot'):
            return
        attrs_dict = {k.lower(): v for k, v in attrs}
        node = TableNode(tag, attrs_dict)
        if self.stack:
            self.stack[-1].children.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag not in ('table', 'tr', 'td', 'th', 'thead', 'tbody', 'tfoot'):
            return
        if len(self.stack) > 1:
            node = self.stack.pop()
            # 收集文本内容
            if self.current_text:
                node.text = ''.join(self.current_text).strip()
                self.current_text = []

    def handle_data(self, data):
        self.current_text.append(data)

    def get_tree(self):
        if self.root.children:
            return self.root.children[0]
        return self.root


class TEDSConfig(Config):
    """APTED 配置：定义节点匹配和开销函数"""

    def rename(self, node1, node2) -> float:
        """重命名开销：节点不同则为1"""
        if node1.name == node2.name:
            return 0.0
        return 1.0

    def children(self, node):
        return node.children if hasattr(node, 'children') else []


def html_to_tree_string(html: str, structure_only: bool = False) -> str:
    """
    将 HTML 表格转换为 APTED tree bracket 格式

    Args:
        html: HTML 字符串
        structure_only: 若True，忽略单元格内容

    Returns:
        bracket_string: APTED bracket 格式字符串
    """
    parser = HTMLTableParser()
    parser.feed(html)
    tree = parser.get_tree()

    def node_to_bracket(node: TableNode) -> str:
        """递归转换为 bracket 格式"""
        if node.tag in ('td', 'th'):
            rs = node.attrs.get('rowspan', '1')
            cs = node.attrs.get('colspan', '1')
            # node label 包含结构信息
            label = f'td rowspan={rs} colspan={cs}'
            if not structure_only and node.text:
                # 包含内容时，用内容作为区分节点的一部分
                text_clean = ' '.join(node.text.split())  # 规范化空白
                label = f'{label} text={text_clean}'
            children_str = ''.join(node_to_bracket(c) for c in node.children)
            return f'{{{label}{children_str}}}'
        else:
            label = node.tag
            children_str = ''.join(node_to_bracket(c) for c in node.children)
            return f'{{{label}{children_str}}}'

    return node_to_bracket(tree)


def compute_teds(pred_html: str, gt_html: str,
                  structure_only: bool = False) -> float:
    """
    计算单个表格的 TEDS 分数

    Args:
        pred_html: 预测的 HTML 字符串
        gt_html: 真实的 HTML 字符串
        structure_only: 若True，计算 TEDS-Struc（忽略内容）

    Returns:
        teds: TEDS 分数 [0, 1]
    """
    try:
        pred_bracket = html_to_tree_string(pred_html, structure_only)
        gt_bracket = html_to_tree_string(gt_html, structure_only)

        pred_tree = Tree.from_text(pred_bracket)
        gt_tree = Tree.from_text(gt_bracket)

        apted = APTED(pred_tree, gt_tree, TEDSConfig())
        edit_dist = apted.compute_edit_distance()

        n_pred = count_nodes(pred_tree)
        n_gt = count_nodes(gt_tree)

        denom = max(n_pred, n_gt)
        if denom == 0:
            return 1.0

        teds = 1.0 - edit_dist / denom
        return max(0.0, teds)

    except Exception:
        return 0.0


def count_nodes(tree) -> int:
    """计算树的节点数"""
    if tree is None:
        return 0
    count = 1
    if hasattr(tree, 'children'):
        for child in tree.children:
            count += count_nodes(child)
    return count


class TEDSEvaluator:
    """
    批量 TEDS 评估器

    对应论文 Section 4.1：
        - TEDS: 完整 TEDS (含内容)
        - TEDS-Struc: 仅结构 TEDS
        - Accuracy: 完全正确预测率（TEDS=1.0的表格比例）

        "Accuracy measures the proportion of tables where both the structure
         and content are fully and correctly recognized"
    """

    def __init__(self, structure_only: bool = False):
        self.structure_only = structure_only
        self.reset()

    def reset(self):
        self.teds_scores = []
        self.teds_struc_scores = []
        self.is_correct = []

    def update(self, pred_html: str, gt_html: str) -> Tuple[float, float]:
        """
        更新评估结果（单个样本）

        Returns:
            (teds, teds_struc) 分数
        """
        teds = compute_teds(pred_html, gt_html, structure_only=False)
        teds_struc = compute_teds(pred_html, gt_html, structure_only=True)

        self.teds_scores.append(teds)
        self.teds_struc_scores.append(teds_struc)
        self.is_correct.append(teds == 1.0)

        return teds, teds_struc

    def compute(self) -> Dict[str, float]:
        """
        计算汇总指标

        Returns:
            {
                'TEDS': float,
                'TEDS-Struc': float,
                'Accuracy': float,
                'n_samples': int
            }
        """
        if not self.teds_scores:
            return {'TEDS': 0.0, 'TEDS-Struc': 0.0, 'Accuracy': 0.0, 'n_samples': 0}

        n = len(self.teds_scores)
        teds = sum(self.teds_scores) / n * 100
        teds_struc = sum(self.teds_struc_scores) / n * 100
        accuracy = sum(self.is_correct) / n * 100

        return {
            'TEDS': teds,
            'TEDS-Struc': teds_struc,
            'Accuracy': accuracy,
            'n_samples': n
        }

    def evaluate_batch(self, pred_htmls: List[str],
                        gt_htmls: List[str]) -> Dict[str, float]:
        """
        批量评估

        Args:
            pred_htmls: 预测 HTML 列表
            gt_htmls: 真实 HTML 列表

        Returns:
            指标字典
        """
        assert len(pred_htmls) == len(gt_htmls)
        for pred, gt in zip(pred_htmls, gt_htmls):
            self.update(pred, gt)
        return self.compute()


def compute_teds_for_dataset(pred_htmls: List[str],
                               gt_htmls: List[str],
                               table_types: Optional[List[str]] = None,
                               n_jobs: int = 4) -> Dict[str, float]:
    """
    对完整数据集批量计算 TEDS 指标

    对应论文 Table 1 的评估结果格式：
        Simple/Complex/All 的 TEDS 和 TEDS-Struc

    Args:
        pred_htmls: 预测 HTML 列表
        gt_htmls: 真实 HTML 列表
        table_types: 每个样本的类型 ('simple' 或 'complex')
        n_jobs: 并行进程数

    Returns:
        {
            'TEDS_all': ..., 'TEDS_simple': ..., 'TEDS_complex': ...,
            'TEDS-Struc_all': ..., 'TEDS-Struc_simple': ..., 'TEDS-Struc_complex': ...,
            'Accuracy': ...
        }
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import numpy as np

    n = len(pred_htmls)

    def _compute_single(args):
        pred, gt = args
        teds = compute_teds(pred, gt, structure_only=False)
        teds_struc = compute_teds(pred, gt, structure_only=True)
        return teds, teds_struc

    print(f"正在计算 {n} 个样本的 TEDS...")

    results = []
    # 使用进程池加速
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_compute_single, (p, g))
                   for p, g in zip(pred_htmls, gt_htmls)]
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 1000 == 0:
                print(f"  已完成 {i+1}/{n}")

    # 整理结果
    teds_all = [r[0] for r in results]
    teds_struc_all = [r[1] for r in results]
    correct = [t == 1.0 for t in teds_all]

    metrics = {
        'TEDS_all': np.mean(teds_all) * 100,
        'TEDS-Struc_all': np.mean(teds_struc_all) * 100,
        'Accuracy': np.mean(correct) * 100,
        'n_samples': n
    }

    # 按简单/复杂表格分类统计
    if table_types:
        simple_idx = [i for i, t in enumerate(table_types) if t == 'simple']
        complex_idx = [i for i, t in enumerate(table_types) if t == 'complex']

        if simple_idx:
            metrics['TEDS_simple'] = np.mean([teds_all[i] for i in simple_idx]) * 100
            metrics['TEDS-Struc_simple'] = np.mean([teds_struc_all[i] for i in simple_idx]) * 100
        if complex_idx:
            metrics['TEDS_complex'] = np.mean([teds_all[i] for i in complex_idx]) * 100
            metrics['TEDS-Struc_complex'] = np.mean([teds_struc_all[i] for i in complex_idx]) * 100

    return metrics


def is_complex_table(html: str) -> bool:
    """
    判断表格是否为复杂表格（含合并单元格）

    对应论文 Section 4.1：
        "The key distinction between [simple and complex] is whether they
         contain row-spanning or column-spanning cells."
    """
    return bool(re.search(r'(rowspan|colspan)=["\']?[2-9]', html, re.IGNORECASE))
