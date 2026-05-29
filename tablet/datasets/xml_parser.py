"""
FinTabNet XML 标注解析工具

解析 XML 结构标注，提取：
- 表格边界框
- 行/列边界框
- 跨行/列单元格边界框
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict, Any


def parse_fintabnet_xml(xml_path: str) -> Dict[str, Any]:
    """
    解析 FinTabNet XML 标注文件

    XML 格式：
        <annotation>
          <filename>...</filename>
          <size><width>W</width><height>H</height></size>
          <object><name>table</name><bndbox>...</bndbox></object>
          <object><name>table row</name><bndbox>...</bndbox></object>
          <object><name>table column</name><bndbox>...</bndbox></object>
          <object><name>table spanning cell</name><bndbox>...</bndbox></object>
          ...
        </annotation>

    Returns:
        {
            'filename': str,
            'width': float,
            'height': float,
            'table_bbox': (xmin, ymin, xmax, ymax) or None,
            'rows': [(ymin, ymax), ...] sorted by ymin,
            'cols': [(xmin, xmax), ...] sorted by xmin,
            'spanning_cells': [{'bbox': (x1,y1,x2,y2), 'type': str}, ...],
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 文件名
    filename_elem = root.find('filename')
    filename = filename_elem.text if filename_elem is not None else ''

    # 图像尺寸
    size = root.find('size')
    if size is not None:
        width = float(size.find('width').text)
        height = float(size.find('height').text)
    else:
        width, height = None, None

    table_bbox = None
    rows = []
    cols = []
    spanning_cells = []

    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        bbox = (xmin, ymin, xmax, ymax)

        if name == 'table':
            table_bbox = bbox
        elif name == 'table row':
            rows.append((ymin, ymax))
        elif name == 'table column':
            cols.append((xmin, xmax))
        elif name in ('table spanning cell', 'table projected row header'):
            spanning_cells.append({'bbox': bbox, 'type': name})

    # 排序
    rows.sort(key=lambda x: x[0])
    cols.sort(key=lambda x: x[0])

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'table_bbox': table_bbox,
        'rows': rows,
        'cols': cols,
        'spanning_cells': spanning_cells,
    }


def get_row_col_split_lines(rows: List[Tuple[float, float]],
                             cols: List[Tuple[float, float]]
                             ) -> Tuple[List[float], List[float]]:
    """
    从行/列标注计算分割线坐标

    分割线定义：
      - 水平分割线：第一行顶边 + 每行底边
      - 垂直分割线：第一列左边 + 每列右边

    Args:
        rows: [(ymin, ymax), ...] 已排序
        cols: [(xmin, xmax), ...] 已排序

    Returns:
        row_lines: 水平分割线 y 坐标，len = num_rows + 1
        col_lines: 垂直分割线 x 坐标，len = num_cols + 1
    """
    row_lines = []
    if rows:
        row_lines.append(rows[0][0])
        for _, ymax in rows:
            row_lines.append(ymax)

    col_lines = []
    if cols:
        col_lines.append(cols[0][0])
        for _, xmax in cols:
            col_lines.append(xmax)

    return row_lines, col_lines
