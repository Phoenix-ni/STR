"""
OTSL (Optimised Table-Structure Language) 工具函数

对应论文 Section 3.2：
    OTSL 定义 4 种 token（无NL）：
    - C: 新单元格（可能有内容，可能是合并单元格的起始）
    - L: 左合并（与左邻单元格合并）
    - U: 上合并（与上邻单元格合并）
    - X: 交叉合并（同时与左和上合并）

将 OTSL 序列转换为 HTML 格式（含 rowspan/colspan 属性）
"""

from typing import List, Tuple, Optional, Dict


# Token 定义
OTSL_C = 'C'   # 新单元格
OTSL_L = 'L'   # 左合并
OTSL_U = 'U'   # 上合并
OTSL_X = 'X'   # 交叉合并

OTSL_TOKENS = [OTSL_C, OTSL_L, OTSL_U, OTSL_X]
OTSL_LABEL_MAP = {t: i for i, t in enumerate(OTSL_TOKENS)}
OTSL_ID2LABEL = {i: t for i, t in enumerate(OTSL_TOKENS)}


def validate_otsl_sequence(sequence: List[str],
                             grid_shape: Tuple[int, int]) -> bool:
    """
    验证 OTSL 序列是否合法

    合法性规则 (OTSL paper [31])：
    1. grid[0][0] 必须为 C
    2. grid[r][0] (首列) 必须为 C 或 U（不能为 L 或 X）
    3. grid[0][c] (首行) 必须为 C 或 L（不能为 U 或 X）
    4. L 左侧必须为 C 或 L
    5. U 上方必须为 C 或 U
    6. X 左侧必须为 U 或 X，上方必须为 L 或 X
    """
    R, C = grid_shape
    if len(sequence) != R * C:
        return False

    grid = [[sequence[r * C + c] for c in range(C)] for r in range(R)]

    for r in range(R):
        for c in range(C):
            t = grid[r][c]
            if t not in OTSL_TOKENS:
                return False
            if r == 0 and c == 0 and t != OTSL_C:
                return False
            if c == 0 and t in (OTSL_L, OTSL_X):
                return False
            if r == 0 and t in (OTSL_U, OTSL_X):
                return False
            if t == OTSL_L:
                if c == 0:
                    return False
                if grid[r][c-1] not in (OTSL_C, OTSL_L):
                    return False
            if t == OTSL_U:
                if r == 0:
                    return False
                if grid[r-1][c] not in (OTSL_C, OTSL_U):
                    return False
            if t == OTSL_X:
                if r == 0 or c == 0:
                    return False
                if grid[r][c-1] not in (OTSL_U, OTSL_X):
                    return False
                if grid[r-1][c] not in (OTSL_L, OTSL_X):
                    return False

    return True


def otsl_sequence_to_spans(sequence: List[str],
                            grid_shape: Tuple[int, int]) -> List[Dict]:
    """
    将 OTSL 序列解析为逻辑单元格列表（含 rowspan/colspan）

    Args:
        sequence: OTSL token 列表，长度 R*C
        grid_shape: (R, C)

    Returns:
        cells: 逻辑单元格列表，每个 dict：
            {
                'row': int,       # 起始行（0-indexed）
                'col': int,       # 起始列（0-indexed）
                'rowspan': int,   # 跨行数
                'colspan': int,   # 跨列数
                'content': str    # 单元格内容（推理时填入）
            }
    """
    R, C = grid_shape
    if len(sequence) == 0:
        return []

    grid = []
    for r in range(R):
        row = []
        for c in range(C):
            idx = r * C + c
            row.append(sequence[idx] if idx < len(sequence) else OTSL_C)
        grid.append(row)

    # 标记每个位置属于哪个逻辑单元格
    # 逻辑单元格由起始 C 位置定义
    cell_map = [[None] * C for _ in range(R)]  # (r,c) -> (start_r, start_c)

    for r in range(R):
        for c in range(C):
            t = grid[r][c]
            if t == OTSL_C:
                cell_map[r][c] = (r, c)
            elif t == OTSL_L:
                # 属于左侧逻辑单元格
                cell_map[r][c] = cell_map[r][c-1] if c > 0 else (r, c)
            elif t == OTSL_U:
                # 属于上方逻辑单元格
                cell_map[r][c] = cell_map[r-1][c] if r > 0 else (r, c)
            elif t == OTSL_X:
                # 属于左上方逻辑单元格
                if r > 0 and c > 0:
                    cell_map[r][c] = cell_map[r-1][c]
                else:
                    cell_map[r][c] = (r, c)

    # 统计每个逻辑单元格的 rowspan/colspan
    cell_info = {}
    for r in range(R):
        for c in range(C):
            start = cell_map[r][c]
            if start is not None:
                if start not in cell_info:
                    cell_info[start] = {'min_r': r, 'max_r': r, 'min_c': c, 'max_c': c}
                else:
                    cell_info[start]['max_r'] = max(cell_info[start]['max_r'], r)
                    cell_info[start]['max_c'] = max(cell_info[start]['max_c'], c)

    # 构建逻辑单元格列表
    cells = []
    for (start_r, start_c), info in cell_info.items():
        if grid[start_r][start_c] != OTSL_C:
            continue
        rowspan = info['max_r'] - info['min_r'] + 1
        colspan = info['max_c'] - info['min_c'] + 1
        cells.append({
            'row': start_r,
            'col': start_c,
            'rowspan': rowspan,
            'colspan': colspan,
            'content': ''
        })

    # 按行列排序
    cells.sort(key=lambda x: (x['row'], x['col']))
    return cells


def otsl_to_html(sequence: List[str],
                  grid_shape: Tuple[int, int],
                  cell_contents: Optional[List[str]] = None) -> str:
    """
    将 OTSL 序列转换为 HTML 表格字符串

    对应论文 Section 3.2（推理输出）：
        "Finally, the OTSL representation is converted into HTML format.
         Then, based on their positions, the OCR-extracted text blocks are
         sequentially placed into their corresponding table cells."

    Args:
        sequence: OTSL token 列表
        grid_shape: (R, C)
        cell_contents: 各逻辑单元格的内容（None则为空）

    Returns:
        html: HTML 表格字符串
    """
    R, C = grid_shape

    if R == 0 or C == 0 or len(sequence) == 0:
        return '<html><body><table></table></body></html>'

    # 解析逻辑单元格
    cells = otsl_sequence_to_spans(sequence, grid_shape)

    # 若提供内容，填入单元格
    if cell_contents:
        for i, cell in enumerate(cells):
            if i < len(cell_contents):
                cell['content'] = cell_contents[i]

    # 构建 HTML
    # 使用逐行渲染，跟踪哪些位置已被 rowspan 覆盖
    occupied = [[False] * C for _ in range(R)]

    # 为每个网格位置建立索引：(r, c) -> 逻辑单元格
    grid_to_cell = {}
    for cell in cells:
        sr, sc = cell['row'], cell['col']
        for dr in range(cell['rowspan']):
            for dc in range(cell['colspan']):
                grid_to_cell[(sr + dr, sc + dc)] = cell

    html_lines = ['<html>', '<body>', '<table>']

    for r in range(R):
        html_lines.append('<tr>')
        for c in range(C):
            if occupied[r][c]:
                continue
            cell = grid_to_cell.get((r, c))
            if cell is None:
                html_lines.append('<td></td>')
                continue

            # 只在起始位置渲染
            if cell['row'] != r or cell['col'] != c:
                continue

            # 标记占用位置
            for dr in range(cell['rowspan']):
                for dc in range(cell['colspan']):
                    if r + dr < R and c + dc < C:
                        occupied[r + dr][c + dc] = True

            # 构建 td 属性
            attrs = []
            if cell['rowspan'] > 1:
                attrs.append(f'rowspan="{cell["rowspan"]}"')
            if cell['colspan'] > 1:
                attrs.append(f'colspan="{cell["colspan"]}"')

            attr_str = (' ' + ' '.join(attrs)) if attrs else ''
            content = cell['content']

            # 对内容中的特殊字符进行转义
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            html_lines.append(f'<td{attr_str}>{content}</td>')

        html_lines.append('</tr>')

    html_lines.extend(['</table>', '</body>', '</html>'])
    return '\n'.join(html_lines)


def otsl_to_html_structure_only(sequence: List[str],
                                  grid_shape: Tuple[int, int]) -> str:
    """
    仅生成结构 HTML（不含内容），用于 TEDS-Struc 评估

    Args:
        sequence: OTSL token 列表
        grid_shape: (R, C)

    Returns:
        html: 无内容的 HTML 表格字符串（用于 TEDS-Struc）
    """
    return otsl_to_html(sequence, grid_shape, cell_contents=None)


def html_to_otsl_sequence(html: str) -> Tuple[List[str], Tuple[int, int]]:
    """
    从 HTML 字符串反向解析为 OTSL 序列（用于评估时的 GT 处理）

    注意：此函数是近似实现，可能不处理所有特殊情况

    Args:
        html: HTML 表格字符串

    Returns:
        (otsl_sequence, grid_shape)
    """
    from html.parser import HTMLParser

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.current_row = None
            self.in_td = False
            self.in_tr = False

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if tag == 'tr':
                self.in_tr = True
                self.current_row = []
            elif tag == 'td' and self.in_tr:
                rowspan = int(attrs_dict.get('rowspan', 1))
                colspan = int(attrs_dict.get('colspan', 1))
                self.current_row.append({'rowspan': rowspan, 'colspan': colspan})
                self.in_td = True

        def handle_endtag(self, tag):
            if tag == 'tr' and self.in_tr:
                if self.current_row is not None:
                    self.rows.append(self.current_row)
                self.in_tr = False
                self.current_row = None
            elif tag == 'td':
                self.in_td = False

    parser = TableParser()
    parser.feed(html)
    rows_data = parser.rows

    if not rows_data:
        return [], (0, 0)

    # 确定网格大小
    # 通过展开所有 rowspan/colspan 来确定 C
    C = 0
    for row in rows_data:
        row_width = sum(cell['colspan'] for cell in row)
        C = max(C, row_width)

    # 展开到完整网格
    expanded_grid = []
    occupied_cols = {}   # 记录被 rowspan 占据的列位置

    for r_idx, row in enumerate(rows_data):
        # 当前行的已有内容（来自之前行的 rowspan）
        grid_row = {}
        if r_idx in occupied_cols:
            for c in occupied_cols[r_idx]:
                grid_row[c] = 'occupied'

        # 填充本行的单元格
        col = 0
        for cell in row:
            while col in grid_row:
                col += 1
            rs, cs = cell['rowspan'], cell['colspan']
            # 标记占用
            for dc in range(cs):
                grid_row[col + dc] = 'cell_start' if dc == 0 else 'cell_cont'
            # 记录后续行的 rowspan 占用
            for dr in range(1, rs):
                future_row = r_idx + dr
                if future_row not in occupied_cols:
                    occupied_cols[future_row] = []
                for dc in range(cs):
                    occupied_cols[future_row].append(col + dc)
            col += cs

        expanded_grid.append(grid_row)

    R = len(expanded_grid)
    if C == 0:
        return [], (0, 0)

    # 转换为 OTSL
    # 重建完整的网格（标记每个位置的 token）
    # 这需要更复杂的逻辑，此处简化
    # 实际使用时建议直接从 XML 标注生成 OTSL
    otsl = []
    for r in range(R):
        for c in range(C):
            otsl.append(OTSL_C)  # 简化

    return otsl, (R, C)
