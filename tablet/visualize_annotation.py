"""
FinTabNet XML标注可视化工具
将表格结构标注绘制到图片上，并生成框线网格

功能:
1. 可视化 XML 标注到图片上
2. 生成框线网格可视化
3. 生成分割掩码 (用于 TABLET Split 模型训练)

使用示例:
    # 可视化单个文件
    python visualize_annotation.py -i image.jpg -x annotation.xml -o output.jpg
    
    # 批量可视化
    python visualize_annotation.py -d ./data/fintabnet -o ./output -n 10
    
    # 生成分割掩码 (用于 Split 模型训练)
    python visualize_annotation.py --generate-masks -d ./data/fintabnet --mask-output ./split_masks
"""

import os
import sys
import argparse
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np


# 定义不同标注类型的颜色 (BGR格式)
COLOR_MAP = {
    'table': (0, 255, 0),           # 绿色 - 整个表格
    'table column header': (0, 0, 255),  # 红色 - 表头
    'table row': (255, 0, 0),       # 蓝色 - 行
    'table column': (255, 255, 0),  # 青色 - 列
    'table cell': (255, 0, 255),    # 紫色 - 单元格
}

# 线宽
LINE_WIDTH = 2


def parse_xml(xml_path):
    """解析XML标注文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    
    # 获取图片尺寸
    size = root.find('size')
    if size is not None:
        width = float(size.find('width').text)
        height = float(size.find('height').text)
    else:
        width, height = None, None
    
    # 解析所有标注对象
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        annotations.append({
            'name': name,
            'bbox': (xmin, ymin, xmax, ymax)
        })
    
    return annotations, (width, height)


def extract_grid_lines(annotations):
    """
    从标注中提取网格线位置
    
    Returns:
        table_bbox: 表格边界框
        row_lines: 水平线y坐标列表 (从上到下)
        col_lines: 垂直线x坐标列表 (从左到右)
    """
    # 获取表格边界
    table_ann = None
    for ann in annotations:
        if ann['name'] == 'table':
            table_ann = ann
            break
    
    if table_ann is None:
        return None, [], []
    
    table_bbox = table_ann['bbox']
    
    # 提取所有行的y坐标
    rows = []
    for ann in annotations:
        if ann['name'] == 'table row':
            _, ymin, _, ymax = ann['bbox']
            rows.append((ymin, ymax))
    
    # 按ymin排序
    rows.sort(key=lambda x: x[0])
    
    # 生成水平线: 第一行的上边界，每行的下边界（即下一行的上边界）
    row_lines = []
    if rows:
        # 第一条线：表格顶部/第一行顶部
        row_lines.append(rows[0][0])
        # 后续线：每行的底部
        for _, ymax in rows:
            row_lines.append(ymax)
    
    # 提取所有列的x坐标
    cols = []
    for ann in annotations:
        if ann['name'] == 'table column':
            xmin, _, xmax, _ = ann['bbox']
            cols.append((xmin, xmax))
    
    # 按xmin排序
    cols.sort(key=lambda x: x[0])
    
    # 生成垂直线
    col_lines = []
    if cols:
        # 第一条线：表格左边界/第一列左边界
        col_lines.append(cols[0][0])
        # 后续线：每列的右边界
        for _, xmax in cols:
            col_lines.append(xmax)
    
    return table_bbox, row_lines, col_lines


def extract_cells_info(annotations):
    """
    从标注中提取单元格信息，用于生成 OTSL 标签
    
    Returns:
        table_bbox: 表格边界框
        rows: 行边界列表 [(ymin, ymax), ...]
        cols: 列边界列表 [(xmin, xmax), ...]
        spanning_cells: 跨越单元格列表 [{'bbox': (xmin, ymin, xmax, ymax), 'type': type}, ...]
    """
    # 获取表格边界
    table_ann = None
    for ann in annotations:
        if ann['name'] == 'table':
            table_ann = ann
            break
    
    if table_ann is None:
        return None, [], [], []
    
    table_bbox = table_ann['bbox']
    
    # 提取所有行
    rows = []
    for ann in annotations:
        if ann['name'] == 'table row':
            _, ymin, _, ymax = ann['bbox']
            rows.append((ymin, ymax))
    rows.sort(key=lambda x: x[0])
    
    # 提取所有列
    cols = []
    for ann in annotations:
        if ann['name'] == 'table column':
            xmin, _, xmax, _ = ann['bbox']
            cols.append((xmin, xmax))
    cols.sort(key=lambda x: x[0])
    
    # 提取跨越单元格 (spanning cell 和 projected row header)
    spanning_cells = []
    for ann in annotations:
        if ann['name'] in ['table spanning cell', 'table projected row header']:
            spanning_cells.append({
                'bbox': ann['bbox'],
                'type': ann['name']
            })
    
    return table_bbox, rows, cols, spanning_cells


def generate_otsl_labels(annotations):
    """
    根据 FinTabNet 标注生成 OTSL 标签
    
    OTSL (Optimised Table-Structure Language) 定义:
    - "C": 新的表格单元格 (可能包含内容)
    - "L": 向左合并的单元格 (与左边单元格合并)
    - "U": 向上合并的单元格 (与上边单元格合并)
    - "X": 交叉单元格 (同时与左边和上边单元格合并)
    
    Args:
        annotations: XML 解析后的标注列表
    
    Returns:
        otsl_sequence: OTSL 标签序列 (按行优先顺序)
        grid_shape: 网格形状 (num_rows, num_cols)
        cell_spans: 每个网格单元的跨度信息 {'row_span': int, 'col_span': int}
    """
    table_bbox, rows, cols, spanning_cells = extract_cells_info(annotations)
    
    if not rows or not cols:
        return [], (0, 0), []
    
    num_rows = len(rows)
    num_cols = len(cols)
    
    # 初始化网格，每个单元格默认为 "C" (独立单元格)
    # grid[i][j] 表示第 i 行第 j 列的单元格类型
    grid = [['C' for _ in range(num_cols)] for _ in range(num_rows)]
    
    # 记录每个单元格的跨度信息
    cell_spans = [[{'row_span': 1, 'col_span': 1} for _ in range(num_cols)] for _ in range(num_rows)]
    
    # 标记被合并的单元格
    # 由于 spanning cell 会覆盖多个网格单元，需要找出哪些单元被合并
    for span_cell in spanning_cells:
        xmin, ymin, xmax, ymax = span_cell['bbox']
        
        # 找出跨越的行和列
        start_row, end_row = None, None
        start_col, end_col = None, None
        
        for i, (r_ymin, r_ymax) in enumerate(rows):
            # 行与跨越单元格有重叠
            if r_ymax > ymin and r_ymin < ymax:
                if start_row is None:
                    start_row = i
                end_row = i + 1  # end_row 是 exclusive
        
        for j, (c_xmin, c_xmax) in enumerate(cols):
            # 列与跨越单元格有重叠
            if c_xmax > xmin and c_xmin < xmax:
                if start_col is None:
                    start_col = j
                end_col = j + 1  # end_col 是 exclusive
        
        if start_row is None or start_col is None:
            continue
        
        # 计算跨度
        row_span = end_row - start_row
        col_span = end_col - start_col
        
        # 更新起始单元格的跨度信息
        cell_spans[start_row][start_col]['row_span'] = row_span
        cell_spans[start_row][start_col]['col_span'] = col_span
        
        # 标记被合并的单元格
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                if i == start_row and j == start_col:
                    # 起始单元格保持 "C"
                    continue
                elif i > start_row and j > start_col:
                    # 同时被行和列合并
                    grid[i][j] = 'X'
                elif i > start_row:
                    # 被上面的单元格合并
                    grid[i][j] = 'U'
                elif j > start_col:
                    # 被左边的单元格合并
                    grid[i][j] = 'L'
    
    # 生成 OTSL 序列 (按行优先顺序)
    otsl_sequence = []
    for i in range(num_rows):
        for j in range(num_cols):
            otsl_sequence.append(grid[i][j])
    
    return otsl_sequence, (num_rows, num_cols), cell_spans


def otsl_to_html(otsl_sequence, grid_shape):
    """
    将 OTSL 序列转换为 HTML 表格字符串
    
    Args:
        otsl_sequence: OTSL 标签序列
        grid_shape: 网格形状 (num_rows, num_cols)
    
    Returns:
        html_string: HTML 表格字符串
    """
    if not otsl_sequence:
        return ""
    
    num_rows, num_cols = grid_shape
    
    html_parts = ['<table>']
    
    idx = 0
    for i in range(num_rows):
        html_parts.append('<tr>')
        for j in range(num_cols):
            if idx >= len(otsl_sequence):
                break
            
            token = otsl_sequence[idx]
            idx += 1
            
            if token == 'C':
                html_parts.append('<td></td>')
            elif token == 'L':
                # 被左边合并，不输出
                pass
            elif token == 'U':
                # 被上面合并，不输出
                pass
            elif token == 'X':
                # 同时被合并，不输出
                pass
        
        html_parts.append('</tr>')
    
    html_parts.append('</table>')
    
    # 注意：这个简化版本没有处理 rowspan 和 colspan 属性
    # 完整版本需要跟踪跨度信息并添加相应属性
    return '\n'.join(html_parts)


def otsl_to_html_with_spans(otsl_sequence, grid_shape, cell_spans):
    """
    将 OTSL 序列和跨度信息转换为完整的 HTML 表格字符串
    
    Args:
        otsl_sequence: OTSL 标签序列
        grid_shape: 网格形状 (num_rows, num_cols)
        cell_spans: 每个网格单元的跨度信息
    
    Returns:
        html_string: HTML 表格字符串
    """
    if not otsl_sequence:
        return ""
    
    num_rows, num_cols = grid_shape
    
    # 跟踪哪些单元格已经被输出（被合并的不需要输出）
    outputted = [[False for _ in range(num_cols)] for _ in range(num_rows)]
    
    html_parts = ['<table>']
    
    idx = 0
    for i in range(num_rows):
        html_parts.append('<tr>')
        for j in range(num_cols):
            if idx >= len(otsl_sequence):
                break
            
            token = otsl_sequence[idx]
            span_info = cell_spans[i][j]
            idx += 1
            
            if token == 'C':
                # 独立单元格或合并单元格的起始
                attrs = []
                if span_info['row_span'] > 1:
                    attrs.append(f"rowspan='{span_info['row_span']}'")
                if span_info['col_span'] > 1:
                    attrs.append(f"colspan='{span_info['col_span']}'")
                
                attr_str = ' ' + ' '.join(attrs) if attrs else ''
                html_parts.append(f'<td{attr_str}></td>')
                
                # 标记被此单元格覆盖的所有位置
                for ri in range(i, i + span_info['row_span']):
                    for cj in range(j, j + span_info['col_span']):
                        if ri < num_rows and cj < num_cols:
                            outputted[ri][cj] = True
            
            elif token in ['L', 'U', 'X']:
                # 被合并的单元格，不输出
                pass
        
        html_parts.append('</tr>')
    
    html_parts.append('</table>')
    
    return '\n'.join(html_parts)


def generate_otsl_from_xml(xml_path):
    """
    从 XML 文件生成 OTSL 标签
    
    Args:
        xml_path: XML 标注文件路径
    
    Returns:
        result: 包含 OTSL 标签和相关信息的字典
    """
    annotations, (xml_width, xml_height) = parse_xml(xml_path)
    
    otsl_sequence, grid_shape, cell_spans = generate_otsl_labels(annotations)
    
    if not otsl_sequence:
        return None
    
    # 生成 HTML
    html_string = otsl_to_html_with_spans(otsl_sequence, grid_shape, cell_spans)
    
    return {
        'otsl_sequence': otsl_sequence,
        'grid_shape': grid_shape,
        'cell_spans': cell_spans,
        'html': html_string,
        'image_size': (xml_width, xml_height)
    }


def lines_to_split_mask(lines, total_length, min_width=5):
    """
    将线条坐标转换为像素级分割掩码
    
    根据 TABLET 的推理后处理形式（split mask → 连通分割区域 → 取中点作为分割线），
    训练/可视化时应将“分割区域”放在真实分割线（边界线）附近的一小段区域上。
    这样从 mask 提取中点时会回到边界线，而不是落在单元格中心。
    
    Args:
        lines: 分割线坐标列表 [p1, p2, p3, ...]，已排序
        total_length: 图像在该方向的尺寸 (H 或 W)
        min_width: 分割区域最小宽度，论文中使用 5 像素
    
    Returns:
        mask: 二值掩码，shape=(total_length,)，1 表示分割区域，0 表示非分割区域
    
    Example:
        >>> lines = [10, 50, 100, 150]
        >>> mask = lines_to_split_mask(lines, 200, min_width=5)
        >>> # mask 在 [28, 33], [73, 78], [123, 128] 位置为 1
    """
    mask = np.zeros(total_length, dtype=np.int64)
    
    if len(lines) < 1:
        return mask
    
    # 所有边界线（含表格外边框：首尾线）
    half_width = max(int(min_width) // 2, 1)
    for p in lines:
        c = int(round(float(p)))
        low = max(0, c - half_width)
        high = min(total_length, c + half_width + 1)
        if high > low:
            mask[low:high] = 1
    
    return mask


def generate_split_masks(xml_path, target_height=None, target_width=None, min_width=5):
    """
    从 XML 标注生成分割掩码，用于 Split 模型训练
    
    Args:
        xml_path: XML 标注文件路径
        target_height: 目标高度，None 则使用原始尺寸
        target_width: 目标宽度，None 则使用原始尺寸
        min_width: 分割区域最小宽度
    
    Returns:
        row_split_mask: 行分割掩码，shape=(H,)，1 表示行分割区域
        col_split_mask: 列分割掩码，shape=(W,)，1 表示列分割区域
        table_bbox: 表格边界框 (xmin, ymin, xmax, ymax)
        original_size: 原始图像尺寸 (width, height)
    """
    # 解析 XML
    annotations, (xml_width, xml_height) = parse_xml(xml_path)
    
    if xml_width is None or xml_height is None:
        raise ValueError(f"无法从 {xml_path} 获取图像尺寸")
    
    # 提取网格线
    table_bbox, row_lines, col_lines = extract_grid_lines(annotations)
    
    if not row_lines or not col_lines:
        return None, None, table_bbox, (xml_width, xml_height)
    
    # 确定目标尺寸
    height = target_height if target_height else int(xml_height)
    width = target_width if target_width else int(xml_width)
    
    # 计算缩放比例
    scale_y = height / xml_height
    scale_x = width / xml_width
    
    # 缩放线条坐标
    row_lines_scaled = [y * scale_y for y in row_lines]
    col_lines_scaled = [x * scale_x for x in col_lines]
    
    # 生成分割掩码
    row_split_mask = lines_to_split_mask(row_lines_scaled, height, min_width)
    col_split_mask = lines_to_split_mask(col_lines_scaled, width, min_width)
    
    # 缩放表格边界框
    if table_bbox is not None:
        table_bbox = (
            table_bbox[0] * scale_x,
            table_bbox[1] * scale_y,
            table_bbox[2] * scale_x,
            table_bbox[3] * scale_y
        )
    
    return row_split_mask, col_split_mask, table_bbox, (xml_width, xml_height)


def visualize_split_mask(row_mask, col_mask, table_bbox=None,
                         img_height=None, img_width=None):
    """
    可视化分割掩码
    
    Args:
        row_mask: 行分割掩码
        col_mask: 列分割掩码
        table_bbox: 表格边界框 (可选)
        img_height: 输出图像高度 (默认使用 row_mask 长度)
        img_width: 输出图像宽度 (默认使用 col_mask 长度)
    
    Returns:
        vis_img: 可视化图像 (BGR)
    """
    height = img_height if img_height else len(row_mask)
    width = img_width if img_width else len(col_mask)
    
    # 创建 RGB 图像
    vis_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 绘制行分割区域 (红色)
    for y, is_split in enumerate(row_mask):
        if is_split:
            vis_img[y, :, 2] = 255  # 红色通道
            vis_img[y, :, 1] = 0
            vis_img[y, :, 0] = 0
    
    # 绘制列分割区域 (绿色)
    for x, is_split in enumerate(col_mask):
        if is_split:
            vis_img[:, x, 1] = 255  # 绿色通道
            vis_img[:, x, 0] = 0
            vis_img[:, x, 2] = 0
    
    # 绘制表格边界框 (蓝色)
    if table_bbox is not None:
        xmin, ymin, xmax, ymax = [int(v) for v in table_bbox]
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    return vis_img


def draw_grid(img_height, img_width, row_lines, col_lines,
              horizontal_color=(0, 0, 255), vertical_color=(0, 255, 0),
              line_width=2, table_bbox=None):
    """
    绘制框线网格
    
    Args:
        img_height: 图片高度
        img_width: 图片宽度
        row_lines: 水平线y标列表
        col_lines: 垂直线x坐标列表
        horizontal_color: 水平线颜色 (BGR) - 默认红色
        vertical_color: 垂直线颜色 (BGR) - 默认绿色
        line_width: 线宽
        table_bbox: 表格边界框 (xmin, ymin, xmax, ymax)，用于限制线条长度
    
    Returns:
        grid_img: 绘制了网格的图片
    """
    # 创建白色背景
    grid_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # 如果提供了表格边界框，使用它来限制线条长度
    if table_bbox is not None:
        tbl_xmin, tbl_ymin, tbl_xmax, tbl_ymax = table_bbox
        tbl_xmin = int(tbl_xmin)
        tbl_ymin = int(tbl_ymin)
        tbl_xmax = int(tbl_xmax)
        tbl_ymax = int(tbl_ymax)
    else:
        tbl_xmin, tbl_ymin = 0, 0
        tbl_xmax, tbl_ymax = img_width, img_height
    
    # 绘制水平线（红色）- 限制在表格左右边界内
    for y in row_lines:
        cv2.line(grid_img, (tbl_xmin, int(y)), (tbl_xmax, int(y)),
                 horizontal_color, line_width)
    
    # 绘制垂直线（绿色）- 限制在表格上下边界内
    for x in col_lines:
        cv2.line(grid_img, (int(x), tbl_ymin), (int(x), tbl_ymax),
                 vertical_color, line_width)
    
    return grid_img


def visualize(image_path, xml_path, output_path=None, show_labels=True, show_grid=True):
    """
    可视化标注到图片上
    
    Args:
        image_path: 图片路径
        xml_path: XML标注文件路径
        output_path: 输出图片路径, None则不保存
        show_labels: 是否显示标签名称
        show_grid: 是否显示框线网格
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # 解析XML
    annotations, (xml_width, xml_height) = parse_xml(xml_path)
    
    # 计算缩放比例(如果XML中的尺寸与图片尺寸不同)
    scale_x = img_width / xml_width if xml_width else 1
    scale_y = img_height / xml_height if xml_height else 1
    
    # 创建图层用于半透明效果
    overlay = img.copy()
    
    # 按类型分组统计
    type_counts = {}
    
    # 绘制每个标注
    for ann in annotations:
        name = ann['name']
        xmin, ymin, xmax, ymax = ann['bbox']
        
        # 缩放坐标
        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)
        
        # 获取颜色
        color = COLOR_MAP.get(name, (128, 128, 128))
        
        # 绘制边界框
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, LINE_WIDTH)
        
        # 填充半透明区域
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, -1)
        
        # 统计
        type_counts[name] = type_counts.get(name, 0) + 1
    
    # 合并原图和标注层(半透明效果)
    alpha = 0.3
    result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    # 再次绘制边界框(不透明)
    for ann in annotations:
        name = ann['name']
        xmin, ymin, xmax, ymax = ann['bbox']
        
        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)
        
        color = COLOR_MAP.get(name, (128, 128, 128))
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, LINE_WIDTH)
        
        # 显示标签
        if show_labels:
            label = name.replace('table ', '')
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            # 计算文字位置
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # 绘制文字背景
            cv2.rectangle(
                result,
                (xmin, ymin - text_height - 5),
                (xmin + text_width, ymin),
                color,
                -1
            )
            
            # 绘制文字
            cv2.putText(
                result, label,
                (xmin, ymin - 3),
                font, font_scale, (255, 255, 255),
                thickness
            )
    
    # 在图片上显示统计信息
    y_offset = 30
    for name, count in sorted(type_counts.items()):
        color = COLOR_MAP.get(name, (128, 128, 128))
        label = f"{name}: {count}"
        cv2.putText(
            result, label,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        y_offset += 20
    
    # 生成框线网格
    if show_grid:
        table_bbox, row_lines, col_lines = extract_grid_lines(annotations)
        
        if row_lines and col_lines:
            # 缩放线条坐标
            row_lines_scaled = [y * scale_y for y in row_lines]
            col_lines_scaled = [x * scale_x for x in col_lines]
            
            # 缩放表格边界框
            table_bbox_scaled = None
            if table_bbox is not None:
                table_bbox_scaled = (
                    table_bbox[0] * scale_x,
                    table_bbox[1] * scale_y,
                    table_bbox[2] * scale_x,
                    table_bbox[3] * scale_y
                )
            
            # 绘制网格
            grid_img = draw_grid(
                img_height, img_width,
                row_lines_scaled, col_lines_scaled,
                horizontal_color=(0, 0, 255),  # 红色 - 水平线
                vertical_color=(0, 255, 0),     # 绿色 - 垂直线
                line_width=2,
                table_bbox=table_bbox_scaled
            )
            
            # 添加统计信息到网格图
            grid_info = f"Grid: {len(row_lines)-1} rows x {len(col_lines)-1} cols"
            cv2.putText(grid_img, grid_info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 添加图例
            cv2.putText(grid_img, "--- Horizontal (Red)", (10, img_height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(grid_img, " |  Vertical (Green)", (10, img_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 将原图和网格图垂直拼接
            result = np.vstack([result, grid_img])
    
    # 显示结果
    cv2.imshow('Annotation Visualization', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"已保存可视化结果到: {output_path}")
    
    return result


def visualize_directory(data_dir, output_dir=None, sample_num=5):
    """
    批量可视化一个目录下的标注
    
    Args:
        data_dir: 数据目录路径 (包含images和val/test/train子目录)
        output_dir: 输出目录
        sample_num: 随机采样数量
    """
    images_dir = os.path.join(data_dir, 'images')
    
    # 查找所有XML文件
    for split in ['val', 'test', 'train']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        xml_files = [f for f in os.listdir(split_dir) if f.endswith('.xml')]
        
        # 随机采样
        if len(xml_files) > sample_num:
            xml_files = random.sample(xml_files, sample_num)
        
        print(f"\n处理 {split} 集, 共 {len(xml_files)} 个样本:")
        
        for xml_file in xml_files:
            xml_path = os.path.join(split_dir, xml_file)
            
            # 对应的图片文件
            img_file = xml_file.replace('.xml', '.jpg')
            img_path = os.path.join(images_dir, img_file)
            
            if not os.path.exists(img_path):
                print(f"  跳过 {xml_file}: 图片不存在")
                continue
            
            # 输出路径
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{split}_{xml_file.replace('.xml', '_vis.jpg')}")
            
            print(f"  可视化: {xml_file}")
            visualize(img_path, xml_path, output_path)


def generate_split_masks_directory(data_dir, output_dir, target_size=960, min_width=5):
    """
    批量生成分割掩码并保存为 .npy 文件
    
    Args:
        data_dir: 数据目录路径 (包含images和val/test/train子目录)
        output_dir: 输出目录
        target_size: 目标尺寸 (高度和宽度相同)
        min_width: 分割区域最小宽度
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        xml_files = [f for f in os.listdir(split_dir) if f.endswith('.xml')]
        
        print(f"\n处理 {split} 集, 共 {len(xml_files)} 个样本:")
        
        for xml_file in xml_files:
            xml_path = os.path.join(split_dir, xml_file)
            
            try:
                # 生成分割掩码
                row_mask, col_mask, table_bbox, original_size = generate_split_masks(
                    xml_path,
                    target_height=target_size,
                    target_width=target_size,
                    min_width=min_width
                )
                
                if row_mask is None or col_mask is None:
                    print(f"  跳过 {xml_file}: 无法提取网格线")
                    continue
                
                # 保存为 .npy 文件
                base_name = xml_file.replace('.xml', '')
                np.save(os.path.join(split_output_dir, f"{base_name}_row_mask.npy"), row_mask)
                np.save(os.path.join(split_output_dir, f"{base_name}_col_mask.npy"), col_mask)
                
                # 保存元数据
                metadata = {
                    'table_bbox': table_bbox,
                    'original_size': original_size,
                    'target_size': target_size,
                    'min_width': min_width
                }
                np.save(os.path.join(split_output_dir, f"{base_name}_metadata.npy"), metadata)
                
            except Exception as e:
                print(f"  错误 {xml_file}: {e}")
        
        print(f"  完成 {split} 集")


def generate_otsl_labels_directory(data_dir, output_dir):
    """
    批量生成 OTSL 标签并保存
    
    Args:
        data_dir: 数据目录路径 (包含images和val/test/train子目录)
        output_dir: 输出目录
    """
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        xml_files = [f for f in os.listdir(split_dir) if f.endswith('.xml')]
        
        print(f"\n处理 {split} 集, 共 {len(xml_files)} 个样本:")
        
        success_count = 0
        for xml_file in xml_files:
            xml_path = os.path.join(split_dir, xml_file)
            
            try:
                # 生成 OTSL 标签
                result = generate_otsl_from_xml(xml_path)
                
                if result is None:
                    continue
                
                # 保存为 JSON 文件
                base_name = xml_file.replace('.xml', '')
                output_path = os.path.join(split_output_dir, f"{base_name}_otsl.json")
                
                # 转换 cell_spans 为可序列化格式
                serializable_spans = []
                for row in result['cell_spans']:
                    row_spans = []
                    for span in row:
                        row_spans.append({
                            'row_span': int(span['row_span']),
                            'col_span': int(span['col_span'])
                        })
                    serializable_spans.append(row_spans)
                
                output_data = {
                    'otsl_sequence': result['otsl_sequence'],
                    'grid_shape': list(result['grid_shape']),
                    'cell_spans': serializable_spans,
                    'html': result['html'],
                    'image_size': list(result['image_size']) if result['image_size'][0] else None
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                
            except Exception as e:
                print(f"  错误 {xml_file}: {e}")
        
        print(f"  完成 {split} 集, 成功生成 {success_count} 个 OTSL 标签")


def main():
    parser = argparse.ArgumentParser(description='FinTabNet XML标注可视化工具')
    parser.add_argument('--image', '-i', type=str, help='图片路径')
    parser.add_argument('--xml', '-x', type=str, help='XML标注文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出图片路径')
    parser.add_argument('--dir', '-d', type=str,
                        default='./data/fintabnet/FinTabNet.c-Structure',
                        help='数据集目录路径')
    parser.add_argument('--sample', '-n', type=int, default=5,
                        help='批量模式下随机采样数量')
    parser.add_argument('--no-labels', action='store_true',
                        help='不显示标签名称')
    parser.add_argument('--no-grid', action='store_true',
                        help='不显示框线网格')
    
    # 分割掩码生成参数
    parser.add_argument('--generate-masks', action='store_true',
                        help='生成分割掩码模式')
    parser.add_argument('--target-size', type=int, default=960,
                        help='目标图像尺寸 (高度和宽度相同)')
    parser.add_argument('--min-width', type=int, default=5,
                        help='分割区域最小宽度')
    parser.add_argument('--mask-output', type=str, default='./split_masks',
                        help='分割掩码输出目录')
    
    # OTSL 标签生成参数
    parser.add_argument('--generate-otsl', action='store_true',
                        help='生成 OTSL 标签模式')
    parser.add_argument('--otsl-output', type=str, default='./otsl_labels',
                        help='OTSL 标签输出目录')
    
    args = parser.parse_args()
    
    if args.generate_masks:
        # 生成分割掩码模式
        generate_split_masks_directory(
            args.dir,
            args.mask_output,
            target_size=args.target_size,
            min_width=args.min_width
        )
    elif args.generate_otsl:
        # 生成 OTSL 标签模式
        generate_otsl_labels_directory(
            args.dir,
            args.otsl_output
        )
    elif args.image and args.xml:
        # 单个文件可视化
        visualize(args.image, args.xml, args.output,
                  not args.no_labels, not args.no_grid)
    else:
        # 批量可视化
        visualize_directory(args.dir, args.output, args.sample)


if __name__ == '__main__':
    main()