# -*- encoding: utf-8 -*-
from typing import List, Dict, Optional, Tuple, Any

class ContextRenderer:
    """Agent渲染引擎结构，复杂处理表格三元组到大模型上下文的可视化映射。"""

    def _split_hierarchical_label(self, text: str) -> Optional[tuple]:
        value = str(text).strip()
        if not value: return None
        for sep in [".", "．"]:
            if sep not in value: continue
            prefix, suffix = value.split(sep, 1)
            prefix, suffix = prefix.strip(), suffix.strip()
            if not prefix or not suffix: continue
            if sep == "." and prefix.replace("-", "").isdigit() and suffix.replace("-", "").isdigit():
                continue
            return prefix, suffix, sep
        return None

    def _detect_groups(self, items: List[str], min_group_size: int = 2) -> List[tuple]:
        split_cache = {item: self._split_hierarchical_label(item) for item in items}
        prefix_counts: Dict[str, int] = {}
        for split_info in split_cache.values():
            if not split_info: continue
            prefix = split_info[0]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        if not any(cnt >= min_group_size for cnt in prefix_counts.values()):
            return []

        groups: List[tuple] = []
        current_label, current_items = None, []
        for item in items:
            split_info = split_cache.get(item)
            group_label = None
            if split_info and prefix_counts.get(split_info[0], 0) >= min_group_size:
                group_label = split_info[0]
            if current_items and group_label != current_label:
                groups.append((current_label, current_items))
                current_items = []
            current_label, group_label = group_label, group_label
            current_items.append(item)
        if current_items:
            groups.append((current_label, current_items))
        return groups if any(label for label, _ in groups) else []

    def _build_group_summary(self, items: List[str]) -> str:
        groups = self._detect_groups(items)
        if not groups: return ""
        parts = []
        for label, members in groups:
            if not label or len(members) < 2: continue
            member_names = []
            for member in members:
                split_info = self._split_hierarchical_label(member)
                member_names.append(split_info[1] if split_info and split_info[0] == label else member)
            parts.append(f"{label}({', '.join(member_names)})")
        return f"表格行分组结构: {'; '.join(parts)}" if parts else ""

    def _escape_md_cell(self, v: Any) -> str:
        s = "" if v is None else str(v)
        return s.replace("\n", " ").replace("\r", " ").replace("|", "/")
        
    def _extract_group_keys(self, group: List[Dict]) -> tuple:
        seen_items, seen_features = set(), set()
        items_out, features_out = [], []
        for g in group:
            if not isinstance(g, dict): continue
            it, ft = g.get("item"), g.get("feature")
            if it and it not in seen_items:
                seen_items.add(it); items_out.append(it)
            if ft and ft not in seen_features:
                seen_features.add(ft); features_out.append(ft)
        return items_out, features_out

    def triplets_to_markdown(
        self,
        triplet_data: Dict,
        pre_context: str = "",
        post_context: str = "",
        selected_items: Optional[List[str]] = None,
        selected_features: Optional[List[str]] = None,
        sub_task_name: str = ""
    ) -> str:
        group = triplet_data.get("group") or []
        shape = triplet_data.get("shape", "")
        # 分组查询特征触发开关
        has_group = "分组" in str(sub_task_name) or "group" in str(sub_task_name)

        real_items, real_features = self._extract_group_keys(group)
        cell_map = {f"{g.get('item')}|||{g.get('feature')}": self._escape_md_cell(g.get('value')) 
                    for g in group if isinstance(g, dict)}

        use_items = selected_items if selected_items else real_items
        use_features = selected_features if selected_features else real_features

        header = ["item"] + [str(f) for f in use_features]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for it in use_items:
            row = [self._escape_md_cell(it)]
            for ft in use_features:
                row.append(cell_map.get(f"{it}|||{ft}", ""))
            lines.append("| " + " | ".join(row) + " |")

        parts = []
        if pre_context: parts.append(pre_context)
        if shape: parts.append(f"Table Size: {shape}")
        if has_group:
            gs = self._build_group_summary(use_items)
            if gs: parts.append(gs)
        parts.append("\n".join(lines))
        if post_context: parts.append(post_context)
        return "\n\n".join(parts)
